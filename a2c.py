import itertools

import gym
import numpy as np
import torch


class RenderWrapper(gym.Wrapper):
    def step(self, action):
        ret = super().step(action)
        super().render()
        return ret


class Mish(torch.nn.Module):
    @staticmethod
    def forward(input_tensor):
        return input_tensor * torch.tanh(torch.nn.functional.softplus(input_tensor))


class Actor(torch.nn.Module):
    def __init__(self, state_dim, n_alternatives):
        """n_alternatives == 1 -> Normal"""
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            Mish(),
            torch.nn.Linear(64, 64),
            Mish(),
            torch.nn.Linear(64, n_alternatives))
        self.n_alternatives = n_alternatives
        if self.n_alternatives == 1:
            self.sigma = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, state):
        dist_params = self.actor(state)
        if self.n_alternatives == 1:
            return torch.distributions.Normal(dist_params.squeeze(), self.sigma)
        else:
            return torch.distributions.Categorical(logits=dist_params)


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            Mish(),
            torch.nn.Linear(64, 64),
            Mish(),
            torch.nn.Linear(64, 1))

    def forward(self, state):
        return self.critic(state).squeeze()


def train_step(states, dists, actions, rewards, dones, act_opt, critic, critic_opt, gamma):
    LAMBDA = 0.92
    act_losses = []
    critic_losses = []
    gae = 0.0
    with torch.no_grad():
        next_value = critic(states[-1])
    for step_id in reversed(range(1, len(states) - 1)):  # TODO without for loop, but I need to build multidimensional dists
        values = critic(states[step_id])
        detached_values = values.detach()
        delta = rewards[step_id] + (1.0 - dones[step_id]) * gamma * next_value - detached_values
        gae = delta + gamma * LAMBDA * (1.0 - dones[step_id]) * gae
        critic_losses.append(torch.nn.functional.mse_loss(values, rewards[step_id] + (1.0 - dones[step_id]) * gamma * next_value))  # TODO TD(lambda)
        log_prob = dists[step_id].log_prob(actions[step_id])
        act_losses.append(-1e-3 * dists[step_id].entropy().mean() - (log_prob * gae).mean())
        next_value = detached_values

    act_opt.zero_grad()
    torch.stack(act_losses).mean().backward()
    act_opt.step()

    critic_opt.zero_grad()
    critic_loss = torch.stack(critic_losses).mean()
    critic_loss.backward()
    critic_opt.step()
    return critic_loss.item()


def main():
    GAMMA = torch.tensor(0.99, dtype=torch.float32)
    # ENV_NAME = 'CartPole-v1'
    ENV_NAME = 'Pendulum-v0'
    envs = gym.vector.async_vector_env.AsyncVectorEnv((lambda: gym.make(ENV_NAME),) * 40)  # 2 + (lambda: RenderWrapper(gym.make(ENV_NAME)),)
    obs_dim = envs.observation_space.shape[1]
    try:
        n_actions = envs.action_space[0].n
    except AttributeError:
        n_actions = 1

    actor = Actor(obs_dim, n_actions)
    act_opt = torch.optim.Adam(actor.parameters(), 1e-3)
    critic = Critic(obs_dim)
    critic_opt = torch.optim.Adam(critic.parameters(), 1e-3)

    current_scores = np.zeros((envs.num_envs,), dtype=np.float64)  # VectorEnv returns as np.float64
    last_scores = []
    losses = []

    states = torch.as_tensor(envs.reset(), dtype=torch.float32)
    mem = [], [], [], [], []  # states, dist, actions, rewards, dones
    for step_id in itertools.count():
        dist = actor(states)
        with torch.no_grad():
            actions = dist.sample()
        if n_actions == 1:
            next_observations, rewards, dones, diagnostic_infos = envs.step(actions[:, None].numpy())  # env needs an extra dim
        else:
            next_observations, rewards, dones, diagnostic_infos = envs.step(actions.numpy())
        next_states = torch.as_tensor(next_observations, dtype=torch.float32)
        tensor_rewards = torch.as_tensor(rewards, dtype=torch.float32)
        tensor_dones = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32)
        mem[0].append(states)
        mem[1].append(dist)
        mem[2].append(actions)
        mem[3].append(tensor_rewards)
        mem[4].append(tensor_dones)

        current_scores += rewards
        number_of_dones = dones.sum()
        if number_of_dones:
            last_scores.extend(current_scores[dones])
            # summary_writer.add_scalar('score', current_scores[dones].sum() / number_of_dones, step_idx)
            # summary_writer.add_scalar('epsilon', epsilon, step_idx)
            current_scores[dones] = 0.0
            if len(last_scores) > 100:
                print(np.mean(last_scores), np.mean(losses))
                # summary_writer.add_scalar('mean score', np.mean(last_scores), step_idx)
                # summary_writer.add_scalar('std score', np.std(last_scores), step_idx)
                last_scores = []
                losses = []

        if len(mem[0]) > 11:
            mem[0].append(next_states)
            critic_loss = train_step(*mem, act_opt, critic, critic_opt, GAMMA)
            losses.append(critic_loss)
            mem = [], [], [], [], []

        states = next_states


if __name__ == '__main__':
    main()
