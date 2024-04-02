import gymnasium as gym
import math
import matplotlib
import numpy
import random
import torch

from collections import deque, namedtuple
from gymnasium import spaces
from itertools import count
from matplotlib import pyplot
from numpy import ndarray
from torch import nn, optim
from torch.nn import functional as F

class SkatingRinkEnv(gym.Env):
    speed = .25
    ang_speed = 1/10 * (2 * numpy.pi)

    end = numpy.array([10, 10])

    state : ndarray
    action_space : spaces.Discrete
    observation_space : spaces.Box

    def __init__(self):
        super(SkatingRinkEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=numpy.array([0, 0, -numpy.pi]), high=numpy.array([100, 100, numpy.pi]), dtype=numpy.float32)

        self.state = numpy.zeros(3)

    def step(self, action : int) -> tuple[ndarray, float, bool, dict[None, None]]:
        sign = action - 1

        self.state = numpy.array([
            self.state[0] + self.speed * numpy.sin(self.state[2]),
            self.state[1] + self.speed * numpy.cos(self.state[2]),
            self.state[2] + self.ang_speed * sign,
        ])

        coords = self.state[[0, 1]]
        done = numpy.sqrt(numpy.sum((coords - self.end) ** 2)) < 2
        reward = 1000 if done else -0.1
        info : dict[None, None] = {}

        return self.state, reward, done, info

    def reset(self) -> ndarray:
        self.state = numpy.zeros(3)
        return self.state

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Trainer:
    env : gym.Env
    model : nn.Module
    target_model : nn.Module
    optimizer : optim.Adam

    eps_start : float
    eps_end : float
    eps_decay : float

    replay_buffer : ReplayBuffer

    def __init__(self, env : gym.Env, model : nn.Module, target_model : nn.Module, optimizer : optim.Adam):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()

        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 500

        self.replay_buffer = ReplayBuffer(10000)

    def eps_by_episode(self, episode : int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * numpy.exp(-1. * episode / self.eps_decay)

    def train_episode(self, epsilon : float) -> None | int:
        batch_size = 64
        gamma = 0.99

        state = self.env.reset()
        episode_rewards = 0

        for e in range(0, 100000):
            # print(state)

            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = self.model(state_tensor).argmax().item()

            next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_rewards += reward

            if len(self.replay_buffer) > batch_size:
                batch = self.replay_buffer.sample(batch_size)
                batch_states_raw, batch_actions_raw, batch_rewards_raw, batch_next_states_raw, batch_dones_raw = zip(*batch)

                batch_states = torch.FloatTensor(numpy.array(batch_states_raw))
                batch_actions = torch.LongTensor(numpy.array(batch_actions_raw))
                batch_rewards = torch.FloatTensor(numpy.array(batch_rewards_raw))
                batch_next_states = torch.FloatTensor(numpy.array(batch_next_states_raw))
                batch_dones = torch.FloatTensor([float(x) for x in numpy.array(batch_dones_raw)])

                current_q = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                next_q = self.model(batch_next_states).max(1)[0]
                expected_q = batch_rewards + gamma * next_q * (1 - batch_dones)

                loss = self.loss_fn(current_q, expected_q.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if done:
                return episode_rewards
        else:
            return None

    def train(self):
        episodes = 1
        target_update = 10

        for episode in range(episodes):
            eps = self.eps_by_episode(episode)
            rewards = self.train_episode(eps)

            if rewards is not None:
                print(f"Episode: {episode}, Total Reward: {rewards}")
            else:
                print("Didn't get to finish!")

        if episode % target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

def main():
    env = SkatingRinkEnv()
    model = DQN(env.observation_space.shape[0], output_dim = env.action_space.n)
    target_model = DQN(env.observation_space.shape[0], output_dim = env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr = .001)

    Trainer(env, model, target_model, optimizer).train()

if __name__ == '__main__':
    main()
