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
from torch import nn, optim, tensor
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkatingRinkEnv(gym.Env):
    speed = .25
    ang_speed = 1/10 * (2 * numpy.pi)

    end = numpy.array([10, 10])

    state : ndarray
    action_space : spaces.Discrete
    observation_space : spaces.Box

    actions = {
        'left': 0,
        'straight': 1,
        'right': 2,
    }

    # phi = direction faced by agent
    state_vars = {
        'y': 0,
        'x': 1,
        'phi': 2,
    }

    def __init__(self):
        super(SkatingRinkEnv, self).__init__()

        # 3 discrete action spaces
        self.action_space = spaces.Discrete(len(self.actions))


        space_range = (-100, 100)
        angle_range = (-numpy.pi, numpy.pi)

        ranges = numpy.array([space_range, space_range, angle_range])
        self.observation_space = spaces.Box(low = ranges[:, 0], high = ranges[:, 1])

        # 3 State spaces (y, x, phi)
        self.state = numpy.zeros(3, dtype = numpy.float32)

    def step(self, action : int) -> tuple[ndarray, float, bool, dict[None, None]]:
        sign = action - 1

        y, x, phi = self.state
        
        # Calculate next state y', x' and phi' using trigonometric functions
        self.state = numpy.array(
            [
                y + self.speed * numpy.sin(phi),
                x + self.speed * numpy.cos(phi),
                phi + self.ang_speed * sign,
            ],
            dtype = numpy.float32
        )

        # Distance of agent towards end goal
        distance = numpy.sqrt(numpy.sum((numpy.array([y, x]) - self.end) ** 2))

        # Calculate termination states
        if distance < 1:
            # Reached endpoint
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
        # Evaluation of model for testing phase only
        self.reset()
        with torch.no_grad():
            for e in range(1, 100 + 1):
                q_values = model(tensor(self.state, device = device).unsqueeze(0).to(device)).squeeze().cpu().numpy()
                action = q_values.argmax()
                _, reward, done, _ = self.step(action)

                print(f'{e:-2d}: {self.state[0]:-3.3f} {self.state[1]:-3.3f} {self.state[2]:-3.3f} -> {action}')
                if done:
                    break
            else:
                print('Never finished :-(')

    def reset(self) -> ndarray:
        self.state = numpy.zeros(3, dtype = numpy.float32)
        return self.state

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1=64, hidden_2=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, output_dim)

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

    def __init__(self, env : gym.Env, model : nn.Module, target_model : nn.Module, optimizer : optim.Adam, batch_size : int, gamma : float, max_steps : int, eps_start : float, eps_end : float, eps_decay : int):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = nn.SmoothL1Loss()

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_steps = max_steps

        self.replay_buffer = ReplayBuffer(10000)

    def eps_by_episode(self, episode : int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * numpy.exp(-1. * episode / self.eps_decay)

    def train_episode(self, epsilon : float) -> None | int:
        batch_size = self.batch_size
        gamma = self.gamma

        state = self.env.reset()
        episode_rewards = 0

        for e in range(0, self.max_steps):
            
            # Regular DQN Training
            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = tensor(state, device = device).unsqueeze(0)
                    action = self.model(state_tensor).argmax().item()

            next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_rewards += reward

            if len(self.replay_buffer) > batch_size:
                batch = self.replay_buffer.sample(batch_size)
                batch_states_raw, batch_actions_raw, batch_rewards_raw, batch_next_states_raw, batch_dones_raw = zip(*batch)

                batch_states = tensor(array(batch_states_raw), dtype = torch.float32, device = device)
                batch_actions = tensor(batch_actions_raw, dtype = torch.int64, device = device)
                batch_rewards = tensor(batch_rewards_raw, device = device)
                batch_next_states = tensor(array(batch_next_states_raw), dtype = torch.float32, device = device)
                batch_dones = tensor([float(x) for x in numpy.array(batch_dones_raw)], device = device)

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

    def train(self, episodes=10):

        for episode in range(1, episodes + 1):
            eps = self.eps_by_episode(episode)
            rewards = self.train_episode(eps)

            print(f"Episode: {episode}, Total Reward: {rewards or 0:g}")

        self.target_model.load_state_dict(self.model.state_dict())
        
    def plot_results(self):
        return

def main():
    env = SkatingRinkEnv()
    lr = 0.00001
    batch_size = 64
    gamma = 0.99
    max_steps = 500
    episodes = 1000
    eps_start = 0.9
    eps_end = 0.005
    eps_decay = 500
    model = DQN(env.observation_space.shape[0], output_dim = env.action_space.n).to(device)
    target_model = DQN(env.observation_space.shape[0], output_dim = env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr = lr)

    Trainer(env, model, target_model, optimizer, batch_size, gamma, max_steps, eps_start, eps_end, eps_decay).train(episodes)
    env.eval(model)

if __name__ == '__main__':
    main()
