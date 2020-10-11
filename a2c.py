import collections
import datetime
import itertools
import random
import time

import gym
import gym.wrappers
import gym.envs.atari
import numpy as np
import torch
from torch.utils import tensorboard


class RenderWrapper(gym.Wrapper):
    def step(self, action):
        ret = super().step(action)
        super().render()
        return ret


class CropGrayScaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (94, 72)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        # observation.shape is (210, 160, 3)
        return observation[5:-17:2, 8:-8:2, :].mean(2, dtype=np.float32) / 255


def breakout():
    return gym.wrappers.FrameStack(CropGrayScaleResizeWrapper(gym.wrappers.TimeLimit(gym.envs.atari.AtariEnv('breakout', obs_type='image',
        frameskip=4, repeat_action_probability=0.25), max_episode_steps=60000)), 4)


def conv(in_features, out_features, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
        # torch.nn.BatchNorm2d(out_features),  # TODO try it
        torch.nn.ReLU())  # TODO leaky relu


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, n_alternatives):
        """n_alternatives == 1 -> Normal"""
        super().__init__()  # TODO MLP backbone
        self.extractor = torch.nn.Sequential(
            conv(state_dim, 32, kernel_size=8, stride=4, padding=4),
            conv(32, 64, kernel_size=4, stride=2, padding=2),
            conv(64, 64),
            torch.nn.Flatten(),
            torch.nn.Linear(8320, 512),
            torch.nn.ReLU())
        self.n_alternatives = n_alternatives
        if self.n_alternatives == 1:
            self.sigma = torch.nn.Parameter(torch.tensor(1.0))

        self.actor = torch.nn.Linear(512, self.n_alternatives)
        self.critic = torch.nn.Linear(512, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), 7e-4)  # TODO weight_decay and then mish

    def forward(self, state, predict_distr=True):
        features = self.extractor(state)
        if predict_distr:
            distr_params = self.actor(features)
            if self.n_alternatives == 1:
                return torch.distributions.Normal(distr_params.squeeze(), self.sigma), self.critic(features).squeeze()
            else:
                return torch.distributions.Categorical(logits=distr_params), self.critic(features).squeeze()
        else:
            return self.critic(features).squeeze()


def train_step(mem, detached_next_values, actor_critic):
    gamma = 0.99
    gamma_lambda = gamma * 0.92
    actor_losses = []
    critic_losses = []
    gae = 0.0
    for distrs, mem_values, actions, rewards, inverted_dones in reversed(mem):
        detached_values = mem_values.detach()
        delta = rewards + gamma * inverted_dones * detached_next_values - detached_values
        gae = delta + gamma_lambda * inverted_dones * gae
        actor_losses.append(-1e-2 * distrs.entropy().mean() - (distrs.log_prob(actions) * gae).mean())
        critic_losses.append(torch.nn.functional.mse_loss(mem_values, gae + detached_values))
        detached_next_values = detached_values

    critic_loss = torch.stack(critic_losses).mean()

    actor_critic.optimizer.zero_grad()
    (torch.stack(actor_losses).mean() + 0.25 * critic_loss).backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
    actor_critic.optimizer.step()
    return critic_loss.item()


def main():
    Entry = collections.namedtuple('Entry', ['distrs', 'values', 'actions', 'rewards', 'inverted_dones'])
    SEED = 0xACE
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    envs = gym.vector.async_vector_env.AsyncVectorEnv((breakout,) * 16)  # 5 + (lambda: RenderWrapper(breakout()),)
    envs.seed(SEED)
    try:
        n_actions = envs.action_space[0].n
    except AttributeError:
        n_actions = 1
    actor_critic = ActorCritic(envs.observation_space.shape[1], n_actions)

    current_scores = np.zeros((envs.num_envs,), dtype=np.float32)
    last_scores = []
    critic_losses = []
    states = torch.as_tensor(envs.reset())
    summary_writer = tensorboard.SummaryWriter(datetime.datetime.now().strftime('logs/%d-%m-%Y %H-%M'))
    t0 = time.perf_counter()
    for step_id in itertools.count():
        mem = []
        for _ in range(5):
            distrs, values = actor_critic(states)
            actions = distrs.sample()
            if n_actions == 1:
                # env needs an extra dim
                next_observations, rewards, dones, diagnostic_infos = envs.step(actions[:, None].cpu().numpy())
            else:
                next_observations, rewards, dones, diagnostic_infos = envs.step(actions.cpu().numpy())
            rewards = rewards.astype(np.float32)  # VectorEnv returns with default dtype which is np.float64
            mem.append(Entry(distrs, values, actions, torch.as_tensor(rewards), torch.as_tensor(~dones)))
            next_states = torch.as_tensor(next_observations)
            states = next_states

            current_scores += rewards
            if dones.any():
                last_scores.extend(current_scores[dones])
                current_scores[dones] = 0.0
                if len(last_scores) > 999:
                    scores = np.mean(last_scores)
                    val_loss = np.mean(critic_losses)
                    summary_writer.add_scalar('Scalars/Score', scores, step_id)
                    summary_writer.add_scalar('Scalars/Value loss', val_loss, step_id)
                    print(step_id, scores, val_loss, time.perf_counter() - t0)
                    last_scores = []
                    critic_losses = []

        with torch.no_grad():
            detached_next_values = actor_critic(next_states, False)
        critic_loss = train_step(mem, detached_next_values, actor_critic)
        critic_losses.append(critic_loss)


if __name__ == '__main__':
    main()
