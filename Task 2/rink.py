import gymnasium as gym
import math
import matplotlib
import random
import numpy
import torch

from collections import deque, namedtuple
from gymnasium import spaces
from itertools import count
from matplotlib import pyplot
from numpy import ndarray, array
from torch import nn, optim, tensor, LongTensor, FloatTensor
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkatingRinkEnv(gym.Env):
    speed = .25
    ang_speed = 1/10 * (2 * numpy.pi)

    end = numpy.array([2, 2])

    state : ndarray

    action_space = spaces.Discrete(3)
    observation_space : spaces.Box

    actions = {
        'left': 0,
        'straight': 1,
        'right': 2,
    }

    state_vars = {
        'y': 0,
        'x': 1,
        'phi': 2,
    }

    def __init__(self):
        super(SkatingRinkEnv, self).__init__()

        space_range = (-100, 100)
        angle_range = (-numpy.pi, numpy.pi)

        ranges = numpy.array([space_range, space_range, angle_range])
        self.observation_space = spaces.Box(low = ranges[:, 0], high = ranges[:, 1])

        self.state = numpy.zeros(3, dtype = numpy.float32)

    @classmethod
    def steps(cls, states : tensor, actions : tensor) -> tensor:
        signs = actions - 1

        ys, xs, phis = states.T
        d_phis = cls.ang_speed * signs
        ds = torch.stack(
            [
                cls.speed * torch.sin(phis + d_phis),
                cls.speed * torch.cos(phis + d_phis),
                d_phis,
            ]
        ).T

        new_states = states + ds

        distances = (new_states[:, 0:2] - tensor(cls.end, device = device)).square().sum(axis = 1).sqrt()

        rewards = torch.where(distances < 1, 10000, -0.1)
        rewards = torch.where(distances >= 10, -10000, rewards)

        dones = torch.where((distances < 1) | (distances >= 100), True, False)

        return new_states, rewards, dones


    def step(self, action : int) -> tuple[ndarray, float, bool, dict[None, None]]:
        sign = action - 1

        y, x, phi = self.state

        self.state = numpy.array(
            [
                y + self.speed * numpy.sin(phi),
                x + self.speed * numpy.cos(phi),
                phi + self.ang_speed * sign,
            ],
            dtype = numpy.float32
        )

        distance = numpy.sqrt(numpy.sum((numpy.array([y, x]) - self.end) ** 2))

        if distance < 1:
            reward = 1000.
            done = True
        elif distance >= 100:
            # Out of bounds
            reward = -1000.
            done = True
        else:
            reward = -0.1
            done = False

        return self.state, reward, done, {}

    def eval(self, model : nn.Module):
        self.reset()
        with torch.no_grad():
            for e in range(1, 101):
                q_values = model(tensor(self.state, device = device).unsqueeze(0).to(device)).squeeze().cpu().numpy()
                action = q_values.argmax()
                _, reward, done, _ = self.step(action)

                dist = ((self.state[0] - 2) ** 2 + (self.state[1] - 2) ** 2) ** (1/2)
                ang = math.atan2(self.state[0] - 2, self.state[1] - 2) / numpy.pi
                print(f'{e:-2d}: [{self.state[0]: 3.3f} {self.state[1]: 3.3f} {self.state[2]: 3.3f}] [{dist: 3.3f} {ang: 3.3f}] -> {action}')
                if done:
                    print('Finished :-)')
                    break
            else:
                print('Never finished :-(')

    def reset(self) -> ndarray:
        self.state = numpy.zeros(3, dtype = numpy.float32)
        return self.state

    @classmethod
    def zeros(cls, batch_size : int) -> tensor:
        return torch.zeros((batch_size, len(cls.state_vars)), device = device)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer:
    env : gym.Env
    model : nn.Module
    target_model : nn.Module
    optimizer : optim.Adam

    eps_start : float
    eps_end : float
    eps_decay : float

    def __init__(self, env : gym.Env, model : nn.Module, target_model : nn.Module, optimizer : optim.Adam):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()

        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 100

    def eps_by_episode(self, episode : int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * numpy.exp(-1. * episode / self.eps_decay)

    def train_episode(self, epsilon : float) -> float:
        batch_size = 100
        actions_size = 1000

        states = SkatingRinkEnv.zeros(batch_size)
        rewards = numpy.zeros(batch_size)

        total_loss = tensor(.0)

        replays : tuple[tensor, tensor, tensor, tensor, tensor]

        total_wins = 0
        total_dones = 0

        for e in range(0, 100):
            with torch.no_grad():
                agents = states.shape[0]
                probs = tensor(numpy.random.choice([False, True], p = [epsilon, 1 - epsilon], size = agents), device = device)
                explorations = torch.randint(0, self.env.action_space.n, (agents,), device = device)
                explotations = self.model(states).argmax(1)

            actions = torch.where(probs, explorations, explotations)
            new_states, rewards, dones = SkatingRinkEnv.steps(states, actions)

            total_wins += torch.sum(dones & (rewards > 1000))
            total_dones += dones.sum()

            tup = (states, actions, new_states, rewards, dones)
            for e in range(replays)
                replays[e] = torch.cat(replays[e], tup[e])[-10000:]

            replays.append((states, actions, new_states, rewards, dones))
            states = new_states[~dones]

            n_replays = len(replays[0])
            used_replays_idx = numpy.random.randint(n_replays, size = actions_size)
            used_replays = [replays[i][x] for x in used_replays_idx for i in len(replays)]
            total_loss += self.commit_gradient(*used_replays)

        return total_loss, total_wins, total_dones

    def commit_gradient(self, states, actions, new_states, rewards, dones):
        gamma = 0.99

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.model(new_states).max(axis = 1)[0]

        expected_q = rewards + gamma * next_q * ~dones

        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q, expected_q)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu()

    def train(self):
        episodes = 100

        for episode in range(1, episodes + 1):
            eps = self.eps_by_episode(episode)
            loss, wins, dones = self.train_episode(eps)

            print(f"Episode: {episode}, Total Loss: {loss:5g}, Total Wins: {wins:5g}, Total Dones: {dones}")

        self.target_model.load_state_dict(self.model.state_dict())

def main():
    env = SkatingRinkEnv()
    model = DQN(env.observation_space.shape[0], output_dim = env.action_space.n).to(device)
    target_model = DQN(env.observation_space.shape[0], output_dim = env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr = .001)

    Trainer(env, model, target_model, optimizer).train()
    env.eval(model)

if __name__ == '__main__':
    main()
