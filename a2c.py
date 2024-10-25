import datetime
import itertools
import os
import time

import cv2
import gymnasium
import ale_py
import numpy as np
import torch
from torch.utils import tensorboard


class ImshowWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.count = 0
        self.limit = 9 * 100

    def observation(self, observation):
        if self.count >= self.limit:
            self.count = 0
            cv2.imshow('', observation)  # TODO show it while the net trains. This will also allow to show multiple observations at a time and draw graphs of and critic loss, actions distribution on an image and state embedding
            key = cv2.pollKey()
            if key != -1:
                key = chr(key)
                if '0' <= key <= '9':
                    self.limit = int(key) * 100
        else:
            self.count += 1
        return observation


class MetaRenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.count = 0
        self.limit = 9 * 100

    def step(self, action):
        observation, reward, done, truncated, diagnostic_info = super().step(action)
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
        return observation, reward, done, truncated, diagnostic_info


class RenderWrapper(gymnasium.Wrapper):
    def step(self, action):
        ret = super().step(action)
        super().render()
        return ret


class CropScaleWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (94, 147)
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        # observation.shape is (210, 160)
        return observation[5:-17:2, 7:-6].astype(np.float32) / 255.0  # TODO without scale


class ScaleWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=0.0, shape=old.shape, dtype=np.float32)

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0


class StopScoreOnLifeLossWrapepr(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_lives = 5

    def step(self, action):
        observation, reward, done, truncated, diagnostic_info = super().step(action)
        if done:
            self.prev_lives = 5
            return observation, reward, True, truncated, {None: False}
        else:
            lives = diagnostic_info['lives']
            if lives < self.prev_lives:
                self.prev_lives = lives
                return observation, reward, False, truncated, {None: False}
            else:
                return observation, reward, False, truncated, {None: True}


class CountTimeLimit(gymnasium.wrappers.TimeLimit):
    def __init__(self, env, max_episode_steps):
        super().__init__(env, max_episode_steps)
        self._elapsed_steps = 0
        self.longest_session = 0

    def reset(self, **kwargs):
        if self.longest_session < self._elapsed_steps:
            self.longest_session = self._elapsed_steps
        return super().reset(**kwargs)

    def close(self):
        print(f'{self.longest_session = }')  # Use this value to adgust max_episode_steps
        super().close()


class InvertDonesWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, truncated, diagnostic_info = super().step(action)
        if done:
            return observation, reward, True, truncated, {None: False}
        return observation, reward, False, truncated, {None: True}


def imenv(show=False):
    envargs = {'game': 'space_invaders', 'mode': None, 'difficulty': None, 'obs_type': 'grayscale', 'frameskip': 5, 'repeat_action_probability': 0.25, 'full_action_space': True, 'render_mode': None}
    if show:
        return gymnasium.wrappers.FrameStackObservation(ImshowWrapper(StopScoreOnLifeLossWrapepr(CropScaleWrapper(CountTimeLimit(ale_py.AtariEnv(
            **envargs), max_episode_steps=50_000)))), 1)
    return gymnasium.wrappers.FrameStackObservation(StopScoreOnLifeLossWrapepr(CropScaleWrapper(CountTimeLimit(ale_py.AtariEnv(
        **envargs), max_episode_steps=50_000))), 1)


def ramenv(show=False):
    envargs = {'game': 'breakout', 'mode': None, 'difficulty': None, 'obs_type': 'ram', 'frameskip': 5, 'repeat_action_probability': 0.25, 'full_action_space': True}
    if show:
        return StopScoreOnLifeLossWrapepr(MetaRenderWrapper(ScaleWrapper(CountTimeLimit(ale_py.AtariEnv(render_mode='rgb_array', **envargs), max_episode_steps=50_000))))
    return StopScoreOnLifeLossWrapepr(ScaleWrapper(CountTimeLimit(ale_py.AtariEnv(**envargs), max_episode_steps=50_000)))


def classicenv(show=False):
    envname = 'CartPole-v1'
    if show:
        return RenderWrapper(InvertDonesWrapper(gymnasium.make(envname)))
    return InvertDonesWrapper(gymnasium.make(envname))


def conv(in_features, out_features, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_features),
        torch.nn.LeakyReLU())  # TODO mish


def fc(in_features, out_features):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, out_features, bias=False),
        torch.nn.BatchNorm1d(out_features),
        torch.nn.LeakyReLU())


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, n_alternatives):
        """n_alternatives == 1 -> Normal"""
        super().__init__()
        self.extractor = torch.nn.Sequential(
            conv(state_dim, 32, kernel_size=8, stride=4, padding=4),
            conv(32, 64, kernel_size=4, stride=2, padding=2),  # TODO 264 channles trains slower but reaches highter res
            conv(64, 64),
            torch.nn.Flatten(),
            fc(15808, 512)) if state_dim == 1 else torch.nn.Sequential(  # TODO mix perfomed action into state, not action
                fc(state_dim, 128),
                fc(128, 128),
                fc(128, 512))
        self.lstm = torch.nn.LSTMCell(512+n_alternatives, 513)  # TODO GRU  # TODO 513->512
        self.actor = torch.nn.Linear(513, n_alternatives)
        self.critic = torch.nn.Linear(513, 1)
        self.n_alternatives = n_alternatives
        if self.n_alternatives == 1:
            self.sigma = torch.nn.Parameter(torch.tensor(1.0))

        for module in self.extractor.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(module.weight, 1e-2)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        torch.nn.init.kaiming_normal_(self.actor.weight, 1e-2)
        if self.actor.bias is not None:
            torch.nn.init.zeros_(self.actor.bias)
        torch.nn.init.kaiming_normal_(self.critic.weight, 1e-2)
        if self.actor.bias is not None:
            torch.nn.init.zeros_(self.critic.bias)

        torch.nn.init.orthogonal_(self.lstm.weight_ih)
        torch.nn.init.zeros_(self.lstm.bias_ih)
        torch.nn.init.orthogonal_(self.lstm.weight_hh)
        torch.nn.init.zeros_(self.lstm.bias_hh)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=7e-4, weight_decay=1e-5)

    def forward(self, state, ax, hx, value_only=False):  # ax[B, 1]
        features = torch.cat([self.extractor(state), ax], dim=1)
        hn = self.lstm(features, hx)
        temporal = hn[0]
        # TODO: Horizon-Aware Value Functions - https://arxiv.org/pdf/1802.10031.pdf
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
    gamma_lambda = gamma * 0.95  # TODO: try lambda 0.5
    actor_losses = []
    entropy_losses = []
    entropys_to_report = []
    critic_losses = []
    gae = 0.0
    for distrs, mem_values, actions, rewards, inverted_dones in reversed(mem):
        detached_values = mem_values.detach()
        delta = rewards + gamma * inverted_dones * detached_next_values - detached_values
        gae = delta + gamma_lambda * inverted_dones * gae
        actor_losses.append((distrs.log_prob(actions) * gae))  # TODO mean here or only globally after this loop
        entropy = distrs.entropy()
        entropy_losses.append(entropy / entropy.detach())  # Adaptive entropy regularization from https://arxiv.org/pdf/2007.02529.pdf
        entropys_to_report.append(entropy.detach())
        critic_losses.append(torch.nn.functional.mse_loss(mem_values, gae + detached_values, reduction='none'))
        detached_next_values = detached_values

    actor_loss = -20.0 * torch.stack(actor_losses).mean()
    entropy_loss = torch.stack(entropy_losses).mean()
    entropy_to_report = torch.stack(entropys_to_report).mean()
    critic_loss = 5.0 * torch.stack(critic_losses).mean()

    actor_critic.optimizer.zero_grad(set_to_none=True)
    (actor_loss - entropy_loss + critic_loss).backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.4)  # TODO: test such big gradients actually exist. This clip should keep policy changes relatively small
    actor_critic.optimizer.step()
    return actor_loss.detach().cpu().numpy(), entropy_to_report.cpu().numpy(), critic_loss.detach().cpu().numpy()  # TODO item() vs numpy()


def randplay(envs):
    current_scores = np.zeros((envs.num_envs,))  # VectorEnv returns with default dtype which is np.float64
    last_scores = []
    envs.reset()
    for step_id in itertools.count():
        next_observations, rewards, dones, truncated, diagnostic_infos = envs.step(envs.action_space.sample())

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
    # TODO: apply code level optimisations from https://arxiv.org/pdf/2005.12729.pdf - ATTRIBUTING SUCCESS IN PROXIMAL POLICY OPTIMIZATION. They are simple
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # TODO try wrappers from stable-baselines3, try EnvPool to speed up
    envs = gymnasium.vector.async_vector_env.AsyncVectorEnv((lambda: imenv(show=True),) + (imenv,) * 47)  # TODO try small number of envs: 4, 8
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
            # one_hot = actions[:, None]  # For Pendulum-v1 with one continuous action
            one_hot = torch.nn.functional.one_hot(actions, num_classes=n_actions)
            if n_actions == 1:
                # env needs an extra dim
                next_observations, rewards, dones, truncated, diagnostic_infos = envs.step(actions[:, None].cpu().numpy())
            else:
                next_observations, rewards, dones, truncated, diagnostic_infos = envs.step(actions.cpu().numpy())
            rewards = rewards.astype(np.float32)  # VectorEnv returns with default dtype which is np.float64
            mem.append((distrs, values, actions, torch.as_tensor(rewards), torch.tensor(diagnostic_infos[None])))
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
                        summary_writer = tensorboard.SummaryWriter(datetime.datetime.now().strftime(f'{os.path.dirname(os.path.abspath(__file__))}/runs/%d%b%H-%M'))
                    step_idk = round(step_id * 0.001)
                    summary_writer.add_scalar('Score', scores, step_idk)
                    summary_writer.add_scalar('Losses/Actor', mean_losses[0], step_idk)
                    summary_writer.add_scalar('Losses/Entropy', mean_losses[1], step_idk)
                    summary_writer.add_scalar('Losses/Critic', mean_losses[2], step_idk)  # TODO report distribution over actions
                    print(f'{color}{step_idk:5,}k {hours:2}:{mins:2} {scores:8,.1f} {mean_losses}')  # TODO tqdm  # TODO customize np print

        with torch.no_grad():
            detached_next_values = actor_critic(next_states, one_hot, prev_h, value_only=True)
        weighted_loss = train_step(mem, detached_next_values, actor_critic)
        prev_h = prev_h[0].detach(), prev_h[1].detach()
        weighted_losses.append(weighted_loss)
        mem *= 0


if __name__ == '__main__':
    main()
