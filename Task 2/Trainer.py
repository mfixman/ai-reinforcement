import gymnasium as gym
import numpy
import torch

from torch import nn, optim, tensor, LongTensor, FloatTensor, BoolTensor
from typing import Any

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    env: gym.Env
    model: nn.Module
    optimizer: optim.Adam

    config: dict[str, Any]

    eps_start: float
    eps_end: float
    eps_decay: float

    def __init__(self, config : dict[str, Any], env: gym.Env, model: nn.Module, optimizer: optim.Adam):
        self.env = env
        self.model = model

        self.config = config

        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()

        self.eps_start = self.config['eps_start']
        self.eps_end = self.config['eps_end']
        self.eps_decay = self.config['eps_decay']

    def eps_by_episode(self, episode: int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * numpy.exp(-1. * episode / self.eps_decay)

    @torch.no_grad()
    def choose_action(self, epsilon: float, states : tensor):
        size = (states.shape[0], )
        probs = tensor(numpy.random.choice([False, True], p = [epsilon, 1 - epsilon], size = size)).to(device)
        explorations = torch.randint(0, self.env.actions_n, size).to(device)
        explotations = self.model(states).argmax(1)
        return torch.where(probs, explotations, explorations)

    def train_episode(self, epsilon: float) -> tuple[float, int, int]:
        batch_size = self.config['batch_size']
        actions_size = self.config['actions_size']
        train_steps = self.config['train_steps']
        buf_multiplier = self.config['buf_multiplier']

        states = self.env.zeros(batch_size)
        rewards = numpy.zeros(batch_size)

        total_loss = tensor(.0)

        replays = (FloatTensor().to(device), LongTensor().to(device), FloatTensor().to(device), FloatTensor().to(device), BoolTensor().to(device))

        total_wins = 0
        total_dones = 0
        for e in range(0, train_steps):
            actions = self.choose_action(epsilon, states)
            new_states, rewards, dones = self.env.steps(states, actions)

            total_wins += torch.sum(dones & (rewards > 1000))
            total_dones += dones.sum()

            tup = (states, actions, new_states, rewards, dones)
            replays = tuple(torch.cat([replays[e], tup[e]])[-1 * buf_multiplier * actions_size:] for e in range(5))

            states = new_states[~dones]

            n_replays = replays[0].shape[0]
            used_replays_idx = numpy.random.randint(n_replays, size = actions_size)
            used_replays = tuple(replays[i][used_replays_idx] for i in range(5))
            total_loss += self.commit_gradient(*used_replays)

            if states.shape[0] == 0:
                break

        return total_loss, total_wins, total_dones

    def commit_gradient(self, states, actions, new_states, rewards, dones):
        gamma = self.config['gamma']

        with torch.no_grad():
            next_q = self.model(new_states).max(axis = 1)[0]
            expected_q = rewards + gamma * next_q * ~dones

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q, expected_q)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu()

    def train(self):
        episodes = self.config['train_episodes']

        for episode in range(1, episodes + 1):
            eps = self.eps_by_episode(episode)
            loss, wins, dones = self.train_episode(eps)

            passes = self.env.eval(self.model, debug = episode % 10 == 0)
            print(f"Episode: {episode:-2d} {'Yes!' if passes else 'Nope'}, Eps = {eps:.2f}, Total Wins: {wins:5g}, Total Dones: {dones:5g}")
