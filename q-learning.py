import gym
import gym.spaces
import gym.wrappers
import numpy as np

import random
import sys

import torch
import torch.nn as nn

import tensorboardX
import datetime

import itertools


class Memory:
    def __init__(self, size=100000, device='cpu'):
        self.size = size
        self.pointer = 0
        self.mem = []
        self.device = device

    def add(self, sarsd):
        if len(self.mem) <= self.size:
            self.mem.append(sarsd)
        else:
            self.mem[self.pointer] = sarsd
            if self.pointer >= self.size:
                self.pointer = 0
            else:
                self.pointer += 1

    def batch(self, batch_size):
        batch = random.sample(self.mem, min(len(self.mem), batch_size))
        batch = [*zip(*batch)]
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


def visualize(checkpoint_name):
    DEVICE = 'cpu'
    env = gym.make('CartPole-v1').env
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape
    action_state_value_func = define_network(state_dim[0], n_actions).to(DEVICE)
    action_state_value_func.load_state_dict(torch.load(checkpoint_name))
    action_state_value_func.eval()
    with torch.no_grad():
        for session_idx in itertools.count():
            observation = env.reset()
            for step_idx in range(1000):
                env.render()
                q_values = action_state_value_func(torch.tensor([observation], dtype=torch.float32, device=DEVICE))
                print(step_idx, q_values)
                _, best_action = q_values.max(1)
                next_observation, reward, done, diagnostic_info = env.step(best_action.item())
                if done:
                    break
                observation = next_observation



def main():
    DEVICE = 'cpu'
    BATCH_SIZE = 1000
    mem = Memory()
    GAMMA = 0.5
    env = gym.make('CartPole-v1').env
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape
    action_state_value_func = define_network(state_dim[0], n_actions).to(DEVICE)
    optimizer = torch.optim.Adam(action_state_value_func.parameters(), lr=0.0001)
    epsilon = 0.4
    summary_writer = tensorboardX.SummaryWriter('logs/{:%d-%m-%Y %H-%M}'.format(datetime.datetime.now()))
    sum_return_G = 0
    for session_idx in itertools.count():
        return_G = 0
        observation = env.reset()
        for step_idx in range(1000):
            # env.render()
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = action_state_value_func(torch.tensor([observation], dtype=torch.float32, device=DEVICE))
                    _, best_action = q_values.max(1)
                    action = best_action.item()
            next_observation, reward, done, diagnostic_info = env.step(action)
            mem.add((observation, action, reward, next_observation, done))

            return_G += reward
            if done:
                break
            observation = next_observation
        sum_return_G += return_G
        states, actions, rewards, next_states, dones = mem.batch(BATCH_SIZE)
        with torch.no_grad():
            next_values, _ = action_state_value_func(next_states).max(1)
            dones *= -1
            dones += 1
            values = torch.tensor(GAMMA, dtype=torch.float32, device=DEVICE) * dones * next_values + rewards
        chosen_q_values = action_state_value_func(states).gather(1, actions.unsqueeze(1))
        loss = torch.nn.functional.mse_loss(chosen_q_values.squeeze(), values)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (session_idx + 1) % 100 == 0:
            print((session_idx + 1) // 100, sum_return_G / 100, epsilon)
            summary_writer.add_scalar('return_G', sum_return_G / 100, session_idx // 100)
            sum_return_G = 0
            epsilon *= 0.999 if epsilon > 1e-3 else epsilon
            with torch.no_grad():
                action_state_value_func.eval()
                sum_return_G_test = 0
                hist = []
                for session_idx_test in range(50):
                    observation = env.reset()
                    for step_idx in range(1000):
                        q_values = action_state_value_func(torch.tensor([observation], dtype=torch.float32,
                                                                        device=DEVICE))
                        _, best_action = q_values.max(1)
                        if 49 == session_idx_test:
                            hist.append(q_values)
                        next_observation, reward, done, diagnostic_info = env.step(best_action.item())
                        sum_return_G_test += reward
                        if done:
                            break
                        observation = next_observation
                # if hist:
                #     print(hist)
                action_state_value_func.train()
            print(sum_return_G_test / 50)
            summary_writer.add_scalar('return_G_Test', sum_return_G_test / 50, session_idx // 100)
            if (session_idx + 1) % 10000 == 0:
                torch.save(action_state_value_func.state_dict(), f'{session_idx // 100}.pt')


if __name__ == '__main__':
    # main()
    visualize(sys.argv[1] + '99.pt')
