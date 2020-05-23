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

import expreplay


class DQN(torch.nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(state_dim, 32, kernel_size=8, stride=4, padding=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.linear1 = torch.nn.Linear(7680, 512)
        self.linear2 = torch.nn.Linear(512, n_actions)

    def forward(self, state):
        c1 = torch.nn.functional.leaky_relu(self.conv3(torch.nn.functional.leaky_relu(self.conv2(torch.nn.functional.leaky_relu(self.conv1(state))))))
        fc1 = F.leaky_relu(self.linear1(c1.view(c1.shape[0], -1)))
        return self.linear2(fc1)


class CropGrayScaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (84, 75)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        return cv2.resize(cv2.cvtColor(observation[-180:, :160], cv2.COLOR_RGB2GRAY),
                          (75, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def main():
    DEVICE = 'cuda'
    ENV_NUM = 4
    BATCH_SIZE = 64
    START_EPSILON = 1.0
    MIN_EPSILON = 0.1
    STOP_EPSILON_DECAY_AT = 250000
    GAMMA = torch.tensor(0.99, dtype=torch.float32, device=DEVICE)
    LEARNING_RATE = 1e-3
    REPLAY_MEMORY_SIZE = 10**6
    # action_state_value_func = define_network(env.observation_space.shape[1], env.action_space[0].n).to(DEVICE)
    action_state_value_func = DQN(4, 4).to(DEVICE)
    target_func = DQN(4, 4).to(DEVICE)  # copy.deepcopy(action_state_value_func)
    target_func.load_state_dict(action_state_value_func.state_dict())
    target_func.eval()
    def predictor(history):
        with torch.no_grad():
            return action_state_value_func(
                torch.tensor(np.moveaxis(history, 3, 1), dtype=torch.float32, device=DEVICE)).cpu()[None, :, :]
    exp_replay = expreplay.ExpReplay(predictor,
                                     lambda: CropGrayScaleResizeWrapper(gym.wrappers.TimeLimit(
                                         gym.envs.atari.AtariEnv('breakout', obs_type='image', frameskip=4,
                                                                 repeat_action_probability=0.25), 60000)),
                                     1,
                                     (84, 75),
                                     BATCH_SIZE,
                                     REPLAY_MEMORY_SIZE, REPLAY_MEMORY_SIZE // 20,
                                     ENV_NUM, 4,
                                     state_dtype=np.float32)
    exp_replay._before_train()
    exp_replay._init_memory()
    # optimizer = torch.optim.Adam(action_state_value_func.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.RMSprop(action_state_value_func.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-02, weight_decay=0.95, momentum=0.95, centered=False)
    exp_replay.exploration = START_EPSILON
    experiment_name = datetime.datetime.now().strftime('logs/%d-%m-%Y %H-%M')
    summary_writer = tensorboardX.SummaryWriter(experiment_name)
    summary_writer.add_hparams({'b': BATCH_SIZE, 'startEps': START_EPSILON, 'minEps': MIN_EPSILON,
        'stopEpsDecayAt': STOP_EPSILON_DECAY_AT, 'gamma': GAMMA.item(), 'lr': LEARNING_RATE,
        'memSize': REPLAY_MEMORY_SIZE}, {}, 'hparams')
    for step_idx, (observations, actions, rewards, dones) in enumerate(exp_replay):
        observations = torch.tensor(np.moveaxis(observations, 3, 1), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions.astype(np.int64), dtype=torch.int64, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).clamp_(-1.0, 1.0)
        dones = torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            next_values, _ = target_func(observations[:, 1:]).max(1)
        values = GAMMA * (1.0 - dones) * next_values + rewards
        optimizer.zero_grad()
        chosen_q_values = action_state_value_func(observations[:, :-1]).gather(1, actions.unsqueeze(1))
        loss = torch.nn.functional.smooth_l1_loss(chosen_q_values.squeeze(), values)
        loss.backward()
        optimizer.step()

        if exp_replay.exploration > MIN_EPSILON:
            exp_replay.exploration -= (START_EPSILON - MIN_EPSILON) / STOP_EPSILON_DECAY_AT

        if step_idx > 0 and step_idx % 5000 == 0:
            target_func.load_state_dict(action_state_value_func.state_dict())  # copy.deepcopy(action_state_value_func)
            mean, max = exp_replay.runner.reset_stats()
            summary_writer.add_scalar('mean score', mean, step_idx)
            summary_writer.add_scalar('max score', max, step_idx)
            summary_writer.add_scalar('epsilon', exp_replay.exploration, step_idx)
            with torch.no_grad():
                summary_writer.add_scalar('value', chosen_q_values.mean(), step_idx)
            summary_writer.add_scalar('loss', loss.item(), step_idx)


if __name__ == '__main__':
    main()
