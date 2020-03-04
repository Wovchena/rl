import gym
from gym.envs import classic_control

import numpy as np

import random

import torch
import torch.nn as nn

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
    return classic_control.CartPoleEnv()


def main():
    DEVICE = 'cpu'
    ENV_NUM = 45  # TODO 1
    BATCH_SIZE = 100  # TODO 1000
    START_EPSILON = 0.3  # TODO 0.4
    MIN_EPSILON = 0.01  # TODO 1e-3
    EPSILON_DECAY = 0.99999  # TODO 0.999 * 1000
    GAMMA = torch.tensor(0.5, dtype=torch.float32, device=DEVICE)
    LEARNING_RATE = 1e-4
    REPLAY_MEMORY_SIZE = 100000
    mem = Memory(REPLAY_MEMORY_SIZE, DEVICE)
    env = gym.vector.async_vector_env.AsyncVectorEnv((lambda: RenderWrapper(env_fn()),)+(env_fn,)*(ENV_NUM-1))
    action_state_value_func = define_network(env.observation_space.shape[1], env.action_space[0].n).to(DEVICE)
    optimizer = torch.optim.Adam(action_state_value_func.parameters(), lr=LEARNING_RATE)
    epsilon = START_EPSILON
    current_total_rewards = np.zeros((env.num_envs,), dtype=np.float64)  # VectorEnv returns as np.float64
    sum_total_rewards = 0.0
    count_sum_total_rewards = 0
    observations = env.reset()
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
        for train_idx in range(2):
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
            dones = 1 - dones
            values = GAMMA * dones * next_values + rewards
        chosen_q_values = action_state_value_func(states).gather(1, actions.unsqueeze(1))
        loss = torch.nn.functional.mse_loss(chosen_q_values.squeeze(), values)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (session_idx + 1) % 100 == 0:
            print((session_idx + 1) // 100, sum_return_G / 100, epsilon)
            summary_writer.add_scalar('return_G', sum_return_G / 100, session_idx // 100)
            sum_return_G = 0
            if epsilon > 0.025:
                epsilon *= 0.999
            with torch.no_grad():
                action_state_value_func.eval()
                sum_return_G_test = 0
                # hist = []
                for session_idx_test in range(30):
                    observation = env.reset()
                    for step_idx in range(1000):
                        q_values = action_state_value_func(torch.tensor([observation], dtype=torch.float32,
                                                                        device=DEVICE))
                        _, best_action = q_values.max(1)
                        # if 29 == session_idx_test:
                        #     hist.append(q_values)
                        next_observation, reward, done, diagnostic_info = env.step(best_action.item())
                        sum_return_G_test += reward
                        if done:
                            break
                        observation = next_observation
                # if hist:
                #     print(hist)
                action_state_value_func.train()
            # print(sum_return_G_test / 30)
            summary_writer.add_scalar('return_G_Test', sum_return_G_test / 30, session_idx // 100)
            if (session_idx + 1) % 2500 == 0:
                torch.save(action_state_value_func.state_dict(), f'{session_idx // 100}.pt')


if __name__ == '__main__':
    main()
    # visualize(sys.argv[1] + '9.pt')
