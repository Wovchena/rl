import datetime
import itertools
import os
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
            key = cv2.pollKey()
            if key != -1:
                key = chr(key)
                if '0' <= key <= '9':
                    self.limit = int(key) * 100
        else:
            self.count += 1
        return observation


class MetaRenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.count = 0
        self.limit = 9 * 100

    def step(self, action):
        observation, reward, done, diagnostic_info = super().step(action)
        if self.count >= self.limit:
            self.count = 0
            cv2.imshow('', diagnostic_info['rgb'][:, :, ::-1])
            key = cv2.pollKey()
            if key != -1:
                key = chr(key)
                if '0' <= key <= '9':
                    self.limit = int(key) * 100
        else:
            self.count += 1
        return observation, reward, done, diagnostic_info


class RenderWrapper(gym.Wrapper):
    def step(self, action):
        ret = super().step(action)
        super().render()
        return ret


class CropScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (94, 147)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        # observation.shape is (210, 160)
        return observation[5:-17:2, 7:-6].astype(np.float32) / 255.0  # TODO without scale


class ScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space
        self.observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=old.shape, dtype=np.float32)

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0


class StopScoreOnLifeLossWrapepr(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_lives = 5

    def step(self, action):
        observation, reward, done, diagnostic_info = super().step(action)
        if done:
            self.prev_lives = 5
            return observation, reward, True, {None: False}
        else:
            lives = diagnostic_info['lives']
            if lives < self.prev_lives:
                self.prev_lives = lives
                return observation, reward, False, {None: False}
            else:
                return observation, reward, False, {None: True}


class CountTimeLimit(gym.wrappers.TimeLimit):
    def __init__(self, env, max_episode_steps):
        super().__init__(env, max_episode_steps)
        self._elapsed_steps = 0
        self.longest_session = 0

    def reset(self, **kwargs):
        if self.longest_session < self._elapsed_steps:
            self.longest_session = self._elapsed_steps
        return super().reset(**kwargs)

    def close(self):
        print(f'{self.longest_session = } ')
        super().close()


class InvertDonesWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, diagnostic_info = super().step(action)
        if done:
            return observation, reward, True, {None: False}
        return observation, reward, False, {None: True}


def imenv(show=False):
    envargs = {'game': 'breakout', 'mode': None, 'difficulty': None, 'obs_type': 'grayscale', 'frameskip': 5, 'repeat_action_probability': 0.25, 'full_action_space': True, 'render_mode': None}
    if show:
        return gym.wrappers.FrameStack(ImshowWrapper(StopScoreOnLifeLossWrapepr(CropScaleWrapper(CountTimeLimit(gym.envs.atari.AtariEnv(
            **envargs), max_episode_steps=50_000)))), 1)
    return gym.wrappers.FrameStack(StopScoreOnLifeLossWrapepr(CropScaleWrapper(CountTimeLimit(gym.envs.atari.AtariEnv(
        **envargs), max_episode_steps=50_000))), 1)


def ramenv(show=False):
    envargs = {'game': 'breakout', 'mode': None, 'difficulty': None, 'obs_type': 'ram', 'frameskip': 5, 'repeat_action_probability': 0.25, 'full_action_space': True}
    if show:
        return StopScoreOnLifeLossWrapepr(MetaRenderWrapper(ScaleWrapper(CountTimeLimit(gym.envs.atari.AtariEnv(render_mode='rgb_array', **envargs), max_episode_steps=50_000))))
    return StopScoreOnLifeLossWrapepr(ScaleWrapper(CountTimeLimit(gym.envs.atari.AtariEnv(**envargs), max_episode_steps=50_000)))


def classicenv(show=False):
    envname = 'CartPole-v1'
    if show:
        return RenderWrapper(InvertDonesWrapper(gym.make(envname)))
    return InvertDonesWrapper(gym.make(envname))


def conv(in_features, out_features, kernel_size=3, stride=1, padding=1):
    # TODO kaiming_uniform_, kaiming_norm_
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
        # torch.nn.BatchNorm2d(out_features),  # TODO try it, but better pick good weght init: https://arxiv.org/abs/1901.09321
        torch.nn.LeakyReLU())


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, n_alternatives):
        """n_alternatives == 1 -> Normal"""
        super().__init__()  # TODO MLP backbone
        self.extractor = torch.nn.Sequential(
            conv(state_dim, 32, kernel_size=8, stride=4, padding=4),
            conv(32, 64, kernel_size=4, stride=2, padding=2),  # TODO 264 channles trains slower but reaches highter res
            conv(64, 64),
            torch.nn.Flatten(),
            torch.nn.Linear(15808, 512),
            torch.nn.LeakyReLU()) if state_dim == 1 else torch.nn.Sequential(
                torch.nn.Linear(state_dim, 128),  # TODO 256
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 512),  # TODO dropout
                torch.nn.LeakyReLU())
        self.lstm = torch.nn.LSTMCell(512+n_alternatives, 513)  # TODO GRU  # TODO 513->512
        self.actor = torch.nn.Linear(513, n_alternatives)
        self.critic = torch.nn.Linear(513, 1)
        self.n_alternatives = n_alternatives
        if self.n_alternatives == 1:
            self.sigma = torch.nn.Parameter(torch.tensor(1.0))

        # From https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/policies.py#L565
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))  # TODO try kaiming_normal_: https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html, xavier, visualize features with PCA
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(param)
        torch.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.critic.weight, gain=1.0)

        self.optimizer = torch.optim.Adam(self.parameters(), 7e-4, weight_decay=1e-5)  # TODO mish  # TODO AdamW

    def forward(self, state, ax, hx, value_only=False):  # ax[B, 1]
        features = torch.cat([self.extractor(state), ax], dim=1)
        hn = self.lstm(features, hx)
        temporal = hn[0]
        if value_only:
            return self.critic(temporal).squeeze()
        else:
            distr_params = self.actor(temporal)
            if self.n_alternatives == 1:
                return torch.distributions.Normal(distr_params.squeeze(), self.sigma), self.critic(temporal).squeeze(), hn
            else:
                return torch.distributions.Categorical(logits=distr_params), self.critic(temporal).squeeze(), hn


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
    critic_loss = 5.0 * torch.stack(critic_losses).mean()

    actor_critic.optimizer.zero_grad(set_to_none=True)
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
                baseline = np.mean(last_scores)
                print(f'{baseline = }')
                return torch.as_tensor(next_observations), baseline


def main():
    # TODO random search that cuts bad params in the beginning
    # TODO assume game continues when max_episode_steps is hit
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # TODO try wrappers from stable-baselines3
    with gym.vector.async_vector_env.AsyncVectorEnv((lambda: imenv(show=True),) + (imenv,) * 39) as envs:
        states, max_mean_score = randplay(envs)
        try:
            n_actions = envs.action_space[0].n
        except AttributeError:
            n_actions = 1
        actor_critic = ActorCritic(envs.observation_space.shape[1], n_actions)
        one_hot = torch.zeros([envs.num_envs, n_actions])
        prev_h = torch.zeros([envs.num_envs, 513]), torch.zeros([envs.num_envs, 513])

        current_scores = np.zeros((envs.num_envs,), dtype=np.float32)
        last_scores = []
        weighted_losses = []
        mem = []
        summary_writer = None
        t0 = time.perf_counter()
        for step_id in itertools.count():
            for _ in range(5):
                distrs, values, prev_h = actor_critic(states, one_hot, prev_h)
                actions = distrs.sample()
                one_hot = torch.nn.functional.one_hot(actions, num_classes=n_actions)
                # one_hot = torch.rand_like(one_hot)
                if n_actions == 1:
                    # env needs an extra dim
                    next_observations, rewards, dones, diagnostic_infos = envs.step(actions[:, None].cpu().numpy())
                else:
                    next_observations, rewards, dones, diagnostic_infos = envs.step(actions.cpu().numpy())
                rewards = rewards.astype(np.float32)  # VectorEnv returns with default dtype which is np.float64
                mem.append((distrs, values, actions, torch.as_tensor(rewards), torch.tensor([alive[None] for alive in diagnostic_infos])))
                next_states = torch.as_tensor(next_observations)
                states = next_states

                current_scores += rewards
                if dones.any():
                    one_hot[dones] = 0
                    inverted_dones = torch.as_tensor((~dones)[:, None])
                    prev_h = prev_h[0] * inverted_dones, prev_h[1] * inverted_dones  # Stop grad for completed envs
                    last_scores.extend(current_scores[dones])
                    current_scores[dones] = 0.0
                    if len(last_scores) > 249:  # TODO Rerpot every 5 mins
                        scores = np.mean(last_scores)
                        mean_losses = np.mean(weighted_losses, axis=0)
                        mins = round((time.perf_counter() - t0) / 60.0)
                        hours, mins = divmod(mins, 60)
                        if max_mean_score < scores:
                            max_mean_score = scores
                            color = '\33[36m'
                        else:
                            color = '\33[m'
                        last_scores *= 0
                        weighted_losses *= 0
                        if summary_writer is None:
                            summary_writer = tensorboard.SummaryWriter(datetime.datetime.now().strftime(f'{os.path.dirname(os.path.abspath(__file__))}/logs/%d-%m-%Y %H-%M'))
                        step_idk = round(step_id * 0.001)
                        summary_writer.add_scalar('Score', scores, step_idk)
                        summary_writer.add_scalar('Losses/Actor', mean_losses[0], step_idk)
                        summary_writer.add_scalar('Losses/Entropy', mean_losses[1], step_idk)
                        summary_writer.add_scalar('Losses/Critic', mean_losses[2], step_idk)  # TODO report distribution over actions
                        print(f'{color}{step_idk:5,}k {hours:2}:{mins:2} {scores:7,.1f} {mean_losses}')  # TODO tqdm

            with torch.no_grad():
                detached_next_values = actor_critic(next_states, one_hot, prev_h, value_only=True)
            weighted_loss = train_step(mem, detached_next_values, actor_critic)
            prev_h = prev_h[0].detach(), prev_h[1].detach()
            weighted_losses.append(weighted_loss)
            mem *= 0


if __name__ == '__main__':
    main()
