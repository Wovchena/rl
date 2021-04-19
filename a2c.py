import datetime
import itertools
import time

import cv2
import gym
import gym.wrappers
import gym.envs.atari
import numpy as np
import torch
from torch.utils import tensorboard


class ImshowWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.count = 0
        self.limit = 9 * 100

    def observation(self, observation):
        if self.count >= self.limit:
            self.count = 0
            cv2.imshow('', observation)  # TODO show it while the net is trained. This will also allow to show multiple observations at a time and draw graphs of and critic loss, actions distribution on an image and state embedding
            key = cv2.waitKey(1)
            if key != -1:
                key = chr(key)
                if '0' <= key <= '9':
                    self.limit = int(key) * 100
        else:
            self.count += 1
        return observation


class CropResizeGrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (94, 147)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        # observation.shape is (210, 160, 3)
        return observation[5:-17:2, 7:-6, :].mean(2, dtype=np.float32) / 255.0


class StopScoreOnLifeLossWrapepr(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_lives = 5

    def step(self, action):
        observation, reward, done, diagnostic_info = super().step(action)
        if done:
            self.prev_lives = 5
            return observation, reward, True, False
        else:
            lives = diagnostic_info['ale.lives']
            if lives < self.prev_lives:
                self.prev_lives = lives
                return observation, reward, False, False
            else:
                return observation, reward, False, True


def create_env(show=False):
    name = 'pong'
    if show:
        return gym.wrappers.FrameStack(ImshowWrapper(StopScoreOnLifeLossWrapepr(CropResizeGrayScaleWrapper(gym.wrappers.TimeLimit(gym.envs.atari.AtariEnv(
            name, obs_type='image', frameskip=3), max_episode_steps=50_000)))), 3)
    return gym.wrappers.FrameStack(StopScoreOnLifeLossWrapepr(CropResizeGrayScaleWrapper(gym.wrappers.TimeLimit(gym.envs.atari.AtariEnv(
        name, obs_type='image', frameskip=3), max_episode_steps=50_000))), 3)


def conv(in_features, out_features, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
        # torch.nn.BatchNorm2d(out_features),  # TODO try it
        torch.nn.LeakyReLU())


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, n_alternatives):
        """n_alternatives == 1 -> Normal"""
        super().__init__()  # TODO MLP backbone
        self.extractor = torch.nn.Sequential(
            conv(state_dim, 32, kernel_size=8, stride=4, padding=4),
            conv(32, 64, kernel_size=4, stride=2, padding=2),
            conv(64, 64),
            torch.nn.Flatten(),
            torch.nn.Linear(15808, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU())
        self.n_alternatives = n_alternatives
        if self.n_alternatives == 1:
            self.sigma = torch.nn.Parameter(torch.tensor(1.0))

        self.actor = torch.nn.Linear(512, self.n_alternatives)
        self.critic = torch.nn.Linear(512, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), 7e-4, weight_decay=1e-5)  # TODO mish

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
    gamma_lambda = gamma * 0.95
    actor_losses = []
    entropy_losses = []
    critic_losses = []
    gae = 0.0
    for distrs, mem_values, actions, rewards, inverted_dones in reversed(mem):
        detached_values = mem_values.detach()
        delta = rewards + gamma * inverted_dones * detached_next_values - detached_values
        gae = delta + gamma_lambda * inverted_dones * gae
        actor_losses.append((distrs.log_prob(actions) * gae))  # TODO mean here or only globally after this loop
        entropy_losses.append(distrs.entropy())
        critic_losses.append(torch.nn.functional.mse_loss(mem_values, gae + detached_values, reduction='none'))
        detached_next_values = detached_values

    actor_loss = -20.0 * torch.stack(actor_losses).mean()
    entropy_loss = torch.stack(entropy_losses).mean()
    critic_loss = torch.stack(critic_losses).mean()

    actor_critic.optimizer.zero_grad()
    (actor_loss - entropy_loss + critic_loss).backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.4)
    actor_critic.optimizer.step()
    return actor_loss.detach().cpu().numpy(), entropy_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()  # TODO item() vs numpy()


def randplay(envs):
    current_scores = np.zeros((envs.num_envs,), dtype=np.float64)  # VectorEnv returns with default dtype which is np.float64
    last_scores = []
    envs.reset()
    for step_id in itertools.count():
        next_observations, rewards, dones, diagnostic_infos = envs.step(envs.action_space.sample())

        current_scores += rewards
        if dones.any():
            last_scores.extend(current_scores[dones])
            current_scores[dones] = 0.0
            if len(last_scores) > 249:
                print(np.mean(last_scores))
                return next_observations

def main():
    # TODO random search that cuts bad params in the beginning
    # TODO report longest game to set more accurate max_episode_steps
    # TODO assume game continues when max_episode_steps is hit
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    with gym.vector.async_vector_env.AsyncVectorEnv((lambda: create_env(show=True),) + (create_env,) * 39) as envs:
        states = torch.as_tensor(randplay(envs))
        try:
            n_actions = envs.action_space[0].n
        except AttributeError:
            n_actions = 1
        actor_critic = ActorCritic(envs.observation_space.shape[1], n_actions)

        current_scores = np.zeros((envs.num_envs,), dtype=np.float32)
        last_scores = []
        max_mean_score = float('-inf')
        weighted_losses = []
        mem = []
        summary_writer = None
        t0 = time.perf_counter()
        for step_id in itertools.count():
            for _ in range(5):
                distrs, values = actor_critic(states)
                actions = distrs.sample()
                if n_actions == 1:
                    # env needs an extra dim
                    next_observations, rewards, dones, diagnostic_infos = envs.step(actions[:, None].cpu().numpy())
                else:
                    next_observations, rewards, dones, diagnostic_infos = envs.step(actions.cpu().numpy())
                rewards = rewards.astype(np.float32)  # VectorEnv returns with default dtype which is np.float64
                mem.append((distrs, values, actions, torch.as_tensor(rewards), torch.tensor(diagnostic_infos)))
                next_states = torch.as_tensor(next_observations)
                states = next_states

                current_scores += rewards
                if dones.any():
                    last_scores.extend(current_scores[dones])
                    current_scores[dones] = 0.0
                    if len(last_scores) > 249:
                        scores = np.mean(last_scores)
                        mean_losses = np.mean(weighted_losses, axis=0)
                        dur = (time.perf_counter() - t0) / 60.0
                        if max_mean_score < scores:
                            max_mean_score = scores
                        last_scores *= 0
                        weighted_losses *= 0
                        if summary_writer is None:
                            summary_writer = tensorboard.SummaryWriter(datetime.datetime.now().strftime('logs/%d-%m-%Y %H-%M'))
                        summary_writer.add_scalar('Score', scores, step_id)
                        summary_writer.add_scalar('Losses/Actor', mean_losses[0], step_id)
                        summary_writer.add_scalar('Losses/Entropy', mean_losses[1], step_id)
                        summary_writer.add_scalar('Losses/Critic', mean_losses[2], step_id)  # TODO report distribution over actions
                        print(step_id, scores, mean_losses, dur, dur / 60.0, max_mean_score)

            with torch.no_grad():
                detached_next_values = actor_critic(next_states, False)
            weighted_loss = train_step(mem, detached_next_values, actor_critic)
            weighted_losses.append(weighted_loss)
            mem *= 0


if __name__ == '__main__':
    main()
