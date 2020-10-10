import datetime
import itertools

import cv2
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
        self.shape = (84, 75)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        return cv2.resize(cv2.cvtColor(observation[-180:, :160], cv2.COLOR_RGB2GRAY),
                          (75, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255  # TODO don't crop score


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
            torch.nn.Linear(7680, 512),
            torch.nn.ReLU())
        self.n_alternatives = n_alternatives
        if self.n_alternatives == 1:
            self.sigma = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.actor = torch.nn.Linear(512, self.n_alternatives)
        self.critic = torch.nn.Linear(512, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), 1e-3)  # TODO 7e-4

    def forward(self, state):
        features = self.extractor(state)  # TODO merge values and dist_params
        value = self.critic(features).squeeze()
        dist_params = self.actor(features)
        if self.n_alternatives == 1:
            return torch.distributions.Normal(dist_params.squeeze(), self.sigma), value
        else:
            return torch.distributions.Categorical(logits=dist_params), value


def train_step(states, dists, actions, rewards, dones, actor_critic, gamma):
    LAMBDA = 0.92
    act_losses = []
    critic_losses = []
    gae = 0.0
    with torch.no_grad():
        _, next_value = actor_critic(states[-1])  # TODO pass values through mem tuple
        next_dones = dones[-1]
    for step_id in reversed(range(len(states) - 1)):  # TODO without for loop, but I need to build multidimensional dists
        _, values = actor_critic(states[step_id])
        detached_values = values.detach()
        delta = rewards[step_id] + (1.0 - next_dones) * gamma * next_value - detached_values
        gae = delta + gamma * LAMBDA * (1.0 - next_dones) * gae
        critic_losses.append(torch.nn.functional.mse_loss(values, gae + detached_values))
        log_prob = dists[step_id].log_prob(actions[step_id])
        act_losses.append(-1e-2 * dists[step_id].entropy().mean() - (log_prob * gae).mean())
        next_value = detached_values
        next_dones = dones[step_id]

    actor_loss = torch.stack(act_losses).mean()
    critic_loss = torch.stack(critic_losses).mean()

    actor_critic.optimizer.zero_grad()
    (actor_loss + 0.25 * critic_loss).backward()
    actor_critic.optimizer.step()
    return critic_loss.item()


def main():
    GAMMA = torch.tensor(0.99, dtype=torch.float32)
    envs = gym.vector.async_vector_env.AsyncVectorEnv((breakout,) * 16)  # 4 + (lambda: RenderWrapper(breakout()),)
    obs_dim = envs.observation_space.shape[1]
    try:
        n_actions = envs.action_space[0].n
    except AttributeError:
        n_actions = 1

    actor_critic = ActorCritic(obs_dim, n_actions)

    current_scores = np.zeros((envs.num_envs,), dtype=np.float64)  # VectorEnv returns as np.float64
    last_scores = []
    losses = []

    states = torch.as_tensor(envs.reset(), dtype=torch.float32)
    prev_dones = torch.zeros((envs.num_envs,), dtype=torch.float32)
    mem = [], [], [], [], []  # states, dist, actions, rewards, dones
    summary_writer = tensorboard.SummaryWriter(datetime.datetime.now().strftime('logs/%d-%m-%Y %H-%M'))
    for step_id in itertools.count():
        dist, _ = actor_critic(states)
        with torch.no_grad():
            actions = dist.sample()
        if n_actions == 1:
            next_observations, rewards, dones, diagnostic_infos = envs.step(actions[:, None].numpy())  # env needs an extra dim
        else:
            next_observations, rewards, dones, diagnostic_infos = envs.step(actions.numpy())
        next_states = torch.as_tensor(next_observations, dtype=torch.float32)
        tensor_rewards = torch.as_tensor(rewards, dtype=torch.float32)
        tensor_dones = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32)  # TODO lazy conversions, revert dones inplace
        mem[0].append(states)
        mem[1].append(dist)
        mem[2].append(actions)
        mem[3].append(tensor_rewards)
        mem[4].append(prev_dones)
        prev_dones = tensor_dones
        states = next_states

        current_scores += rewards
        if dones.any():
            last_scores.extend(current_scores[dones])
            current_scores[dones] = 0.0
            if len(last_scores) > 999:
                scores = np.mean(last_scores)
                val_loss = np.mean(losses)
                summary_writer.add_scalar('Scalars/Score', scores, step_id)
                summary_writer.add_scalar('Scalars/Value loss', val_loss, step_id)
                print(step_id, scores, val_loss)
                last_scores = []
                losses = []

        if len(mem[0]) > 5:
            mem[0].append(next_states)  # TODO may be pass them as separate arguments
            mem[4].append(tensor_dones)
            critic_loss = train_step(*mem, actor_critic, GAMMA)
            losses.append(critic_loss)
            mem = [], [], [], [], []


if __name__ == '__main__':
    main()
