import gym
from gym.envs import classic_control
import gym.wrappers
import gym.envs.atari

import numpy as np

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorboardX
import datetime

import itertools


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


def conv(in_features, out_features):
    return torch.nn.Sequential(torch.nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(out_features), torch.nn.ReLU())


class ConvQN(torch.nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.conv1 = conv(state_dim, 32)
        self.conv2 = conv(32, 32)
        self.conv3 = conv(32, 32)
        self.conv4 = conv(32, 64)
        self.conv5 = conv(64, 64)
        self.conv6 = conv(64, 64)
        self.linear1 = torch.nn.Linear(537600, 512)
        self.linear2 = torch.nn.Linear(512, n_actions)

    def forward(self, state):
        p1 = self.conv6(self.conv5(self.conv4(torch.nn.functional.max_pool2d(self.conv3(self.conv2(self.conv1(state))), kernel_size=2))))
        fc1 = F.relu(self.linear1(p1.view(p1.shape[0], -1)))
        return self.linear2(fc1)


def visualize(checkpoint_name):
    DEVICE = 'cpu'
    env = classic_control.CartPoleEnv()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape
    action_state_value_func = define_network(state_dim[0], n_actions).to(DEVICE)
    action_state_value_func.load_state_dict(torch.load(checkpoint_name))
    action_state_value_func.eval()
    with torch.no_grad():
        for session_idx in itertools.count():
            observation = env.reset()
            for step_idx in range(2000):
                env.render()
                if random.random() < 0.:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = action_state_value_func(torch.tensor([observation], dtype=torch.float32, device=DEVICE))
                        print(step_idx, q_values)
                        _, best_action = q_values.max(1)
                        action = best_action.item()
                next_observation, reward, done, diagnostic_info = env.step(action)
                if done:
                    break
                observation = next_observation


class RenderWrapper(gym.Wrapper):
    def step(self, action):
        super().render()
        return super().step(action)


def env_fn():
    #return gym.wrappers.FlattenObservation(gym.wrappers.FrameStack(gym.wrappers.TimeLimit(classic_control.CartPoleEnv(), 1000), 3))
    #return gym.wrappers.TimeLimit(classic_control.CartPoleEnv(), 1000)
    return gym.wrappers.FrameStack(gym.wrappers.TimeLimit(gym.envs.atari.AtariEnv('breakout', obs_type='image', frameskip=1, repeat_action_probability=0.25), 1000), 3)


def main():
    DEVICE = 'cuda'
    ENV_NUM = 10 
    BATCH_SIZE = 30 
    START_EPSILON = 0.5
    MIN_EPSILON = 0.001
    EPSILON_DECAY = 0.9999
    GAMMA = torch.tensor(0.5, dtype=torch.float32, device=DEVICE)
    LEARNING_RATE = 1e-4
    REPLAY_MEMORY_SIZE = 5000
    mem = Memory(REPLAY_MEMORY_SIZE, DEVICE)
    env = gym.vector.async_vector_env.AsyncVectorEnv((env_fn,)*ENV_NUM)
    # action_state_value_func = define_network(env.observation_space.shape[1], env.action_space[0].n).to(DEVICE)
    action_state_value_func = ConvQN(9, env.action_space[0].n).to(DEVICE)
    optimizer = torch.optim.Adam(action_state_value_func.parameters(), lr=LEARNING_RATE)
    epsilon = START_EPSILON
    current_total_rewards = np.zeros((env.num_envs,), dtype=np.float64)  # VectorEnv returns as np.float64
    sum_total_rewards = 0.0
    count_sum_total_rewards = 0
    observations = env.reset()
    observations = observations.transpose(0, 2, 3, 1, 4).reshape((observations.shape[0], observations.shape[2], observations.shape[3], -1))  # (20, 3, 210, 160, 3)->(20, 210, 160, 9)
    observations = observations.transpose(0, 3, 1, 2)
    experiment_Name = 'logs/{:%d-%m-%Y %H-%M}'.format(datetime.datetime.now())
    hparam_dict = {'b': BATCH_SIZE, 'startEps': START_EPSILON, 'minEps': MIN_EPSILON, 'epsDecay': EPSILON_DECAY,
                   'gamma': GAMMA.item(), 'lr': LEARNING_RATE, 'memSize': REPLAY_MEMORY_SIZE}
    for step_idx in itertools.count():
        if random.random() < epsilon:
            actions = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = action_state_value_func(torch.tensor(observations, dtype=torch.float32, device=DEVICE))
            actions = q_values.argmax(1).cpu().numpy()  # gym doesn't know PyTorch
        next_observations, rewards, dones, diagnostic_infos = env.step(actions)
        next_observations = next_observations.transpose(0, 2, 3, 1, 4).reshape((next_observations.shape[0], next_observations.shape[2], next_observations.shape[3], -1))  # (20, 3, 210, 160, 3)->(20, 210, 160, 9)
        next_observations = next_observations.transpose(0, 3, 1, 2)
        mem.add(observations, actions, rewards, next_observations, dones)

        # record stats
        current_total_rewards += rewards
        number_of_dones = dones.sum()
        if number_of_dones:  # sum_total_rewards += current_total_rewards[dones].sum() will result in nan otherwise
            sum_total_rewards += current_total_rewards[dones].sum()
            count_sum_total_rewards += number_of_dones
            current_total_rewards[dones] = 0.0
        if step_idx != 0 and count_sum_total_rewards != 0 and step_idx % 100 == 0:
            with tensorboardX.SummaryWriter(experiment_Name) as summary_writer:
                summary_writer.add_hparams(hparam_dict=hparam_dict,
                                           metric_dict={'totalReward': sum_total_rewards/count_sum_total_rewards,
                                                        'epsilon': epsilon},
                                           name='a', global_step=step_idx//100)
            sum_total_rewards = 0.0
            count_sum_total_rewards = 0

        # finalize step
        observations = next_observations
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY

        # train
        for train_idx in range(1):
            states, actions, rewards, next_states, dones = mem.batch(BATCH_SIZE)
            with torch.no_grad():
                next_values, _ = action_state_value_func(next_states).max(1)
            dones = 1 - dones
            values = GAMMA * dones * next_values + rewards
            chosen_q_values = action_state_value_func(states).gather(1, actions.unsqueeze(1))
            loss = torch.nn.functional.mse_loss(chosen_q_values.squeeze(), values)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == '__main__':
    main()
    # visualize(sys.argv[1] + '9.pt')
