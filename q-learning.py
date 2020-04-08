import datetime
import itertools
import random

import cv2

import gym
import gym.envs.atari
# import gym.envs.classic_control
import gym.wrappers

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorboardX


class Memory:
    def __init__(self, size, device):
        self.size = size
        self.pointer = 0
        self.mem = []
        self.device = device

    def add(self, states, actions, rewards, next_states, dones):
        for idx in range(len(states)):
            if len(self.mem) <= self.size:
                self.mem.append((states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]))
            else:
                self.mem[self.pointer] = (states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])
                if self.pointer >= self.size:
                    self.pointer = 0
                else:
                    self.pointer += 1

    def batch(self, batch_size):
        batch = random.sample(self.mem, min(len(self.mem), batch_size))
        batch = tuple(zip(*batch))
        states = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch[1], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones


class Swish(torch.nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)


def define_network(state_dim, n_actions):
    return nn.Sequential(
        nn.Linear(state_dim, 80),
        Swish(),
        nn.Linear(80, 50),
        Swish(),
        nn.Linear(50, n_actions))


def conv(in_features, out_features, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(out_features), Swish())


class DQN(torch.nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        # like https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf but Swish, resize not to square
        self.conv1 = conv(state_dim, 32, kernel_size=8, stride=4, padding=4)
        self.conv2 = conv(32, 64, kernel_size=4, stride=2, padding=2)
        self.conv3 = conv(64, 64)  # TODO: trainable leaky RelU, no batchNorm
        self.linear1 = torch.nn.Linear(7680, 512)
        self.linear2 = torch.nn.Linear(512, n_actions)

    def forward(self, state):
        c1 = self.conv3(self.conv2(self.conv1(state)))
        fc1 = F.relu(self.linear1(c1.view(c1.shape[0], -1)))
        return self.linear2(fc1)


class RenderWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, diagnostic_info = super().step(action)
        # cv2.imshow("kek", observation[2])
        # cv2.waitKey(1)
        super().render()
        return observation, reward, done, diagnostic_info


class CropGrayScaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (84, 75, 1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        return cv2.resize(cv2.cvtColor(observation[-180:, :160], cv2.COLOR_RGB2GRAY),
                          (75, 84), interpolation=cv2.INTER_AREA)[:, :, None]


class SqueezeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 75), dtype=np.uint8)

    def observation(self, observation):
        return np.squeeze(observation, 3)


def env_fn():
    # return gym.wrappers.FlattenObservation(gym.wrappers.FrameStack(gym.wrappers.TimeLimit(gym.envs.classic_control.CartPoleEnv(), 1000), 3))
    return SqueezeWrapper(
        gym.wrappers.FrameStack(
            CropGrayScaleResizeWrapper(
                gym.wrappers.TimeLimit(
                    gym.envs.atari.AtariEnv('breakout', obs_type='image', frameskip=4, repeat_action_probability=0.25),
                    60000),
            ),
            4)
        )


def train_step(action_state_value_func, mem, batch_size, gamma, optimizer):
    optimizer.zero_grad()
    states, actions, rewards, next_states, dones = mem.batch(batch_size)
    with torch.no_grad():
        next_values, _ = action_state_value_func(next_states).max(1)
    values = gamma * (1.0 - dones) * next_values + rewards
    chosen_q_values = action_state_value_func(states).gather(1, actions.unsqueeze(1))
    loss = torch.nn.functional.mse_loss(chosen_q_values.squeeze(), values)
    loss.backward()
    optimizer.step()


def main():
    # TODO report predicted q-values, mem could manage frame stacking
    DEVICE = 'cuda'
    ENV_NUM = 4
    BATCH_SIZE = 64
    START_EPSILON = 1.0
    MIN_EPSILON = 0.1
    STOP_EPSILON_DECAY_AT = 10**6
    GAMMA = torch.tensor(0.99, dtype=torch.float32, device=DEVICE)
    LEARNING_RATE = 1e-3
    REPLAY_MEMORY_SIZE = 5000  # TODO: 10**6
    mem = Memory(REPLAY_MEMORY_SIZE, DEVICE)
    env = gym.vector.async_vector_env.AsyncVectorEnv((lambda: RenderWrapper(env_fn()),)+(env_fn,)*(ENV_NUM-1))
    # action_state_value_func = define_network(env.observation_space.shape[1], env.action_space[0].n).to(DEVICE)
    action_state_value_func = DQN(4, env.action_space[0].n).to(DEVICE)
    optimizer = torch.optim.Adam(action_state_value_func.parameters(), lr=LEARNING_RATE)
    epsilon = START_EPSILON
    current_scores = np.zeros((env.num_envs,), dtype=np.float64)  # VectorEnv returns as np.float64
    rewards = np.zeros((env.num_envs,), dtype=np.float64)
    dones = np.zeros((env.num_envs,), dtype=np.bool)
    last_scores = []
    observations = env.reset()
    experiment_name = datetime.datetime.now().strftime('logs/%d-%m-%Y %H-%M')

    for step_idx in range(REPLAY_MEMORY_SIZE // 20):
        actions = env.action_space.sample()
        next_observations, rewards, dones, diagnostic_infos = env.step(actions)
        mem.add(observations, actions, rewards, next_observations, dones)
        observations = next_observations

    summary_writer = tensorboardX.SummaryWriter(experiment_name)
    summary_writer.add_hparams({'b': BATCH_SIZE, 'startEps': START_EPSILON, 'minEps': MIN_EPSILON,
        'stopEpsDecayAt': STOP_EPSILON_DECAY_AT, 'gamma': GAMMA.item(), 'lr': LEARNING_RATE,
        'memSize': REPLAY_MEMORY_SIZE}, {}, 'hparams')
    for step_idx in itertools.count():
        if random.random() < epsilon:
            actions = env.action_space.sample()
        else:
            with torch.no_grad():
                actions = action_state_value_func(torch.tensor(observations, dtype=torch.float32, device=DEVICE))
            actions = actions.argmax(1).cpu().numpy()
        env.step_async(actions)

        train_step(action_state_value_func, mem, BATCH_SIZE, GAMMA, optimizer)

        # record stats
        current_scores += rewards
        number_of_dones = dones.sum()
        if number_of_dones:
            last_scores.extend(current_scores[dones])
            summary_writer.add_scalar('score', current_scores[dones].sum() / number_of_dones, step_idx)
            summary_writer.add_scalar('epsilon', epsilon, step_idx)
            if len(last_scores) > 100:
                summary_writer.add_scalar('mean score', np.mean(last_scores), step_idx)
                summary_writer.add_scalar('std score', np.std(last_scores), step_idx)
                last_scores = []
            current_scores[dones] = 0.0

        if epsilon > MIN_EPSILON:
            epsilon -= (START_EPSILON - MIN_EPSILON) / STOP_EPSILON_DECAY_AT

        next_observations, rewards, dones, diagnostic_infos = env.step_wait()
        mem.add(observations, actions, rewards, next_observations, dones)
        observations = next_observations


if __name__ == '__main__':
    main()
