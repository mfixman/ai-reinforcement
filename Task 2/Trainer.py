import gymnasium as gym
import numpy
import torch

from torch import nn, optim, tensor, LongTensor, FloatTensor, BoolTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    env: gym.Env
    model: nn.Module
    target_model: nn.Module
    optimizer: optim.Adam

    eps_start: float
    eps_end: float
    eps_decay: float

    def __init__(self, env: gym.Env, model: nn.Module, target_model: nn.Module, optimizer: optim.Adam):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()

        self.eps_start = 1.0
        self.eps_end = 0
        self.eps_decay = 100

    def eps_by_episode(self, episode: int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * numpy.exp(-1. * episode / self.eps_decay)

    def train_episode(self, epsilon: float) -> float:
        batch_size = 10000
        actions_size = 100000

        states = self.env.zeros(batch_size)
        rewards = numpy.zeros(batch_size)

        total_loss = tensor(.0)

        replays: tuple[tensor, tensor, tensor, tensor, tensor]
        replays = (FloatTensor().to(device), LongTensor().to(device), FloatTensor().to(device), FloatTensor().to(device), BoolTensor().to(device))

        total_wins = 0
        total_dones = 0

        for e in range(0, 100):
            with torch.no_grad():
                agents = states.shape[0]
                probs = tensor(numpy.random.choice([False, True], p = [epsilon, 1 - epsilon], size = agents), device = device)
                explorations = torch.randint(0, self.env.action_space.n, (agents,), device = device)
                explotations = self.model(states).argmax(1)

            actions = torch.where(probs, explotations, explorations)

            new_states, rewards, dones = self.env.steps(states, actions)

            total_wins += torch.sum(dones & (rewards > 1000))
            total_dones += dones.sum()

            tup = (states, actions, new_states, rewards, dones)
            replays = tuple(torch.cat([replays[e], tup[e]])[-100 * actions_size:] for e in range(5))

            states = new_states[~dones]

            n_replays = replays[0].shape[0]
            used_replays_idx = numpy.random.randint(n_replays, size = actions_size)
            used_replays = tuple(replays[i][used_replays_idx] for i in range(5))
            total_loss += self.commit_gradient(*used_replays)

            if states.shape[0] == 0:
                break

        return total_loss, total_wins, total_dones

    def commit_gradient(self, states, actions, new_states, rewards, dones):
        gamma = 0.99

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
        episodes = 100

        for episode in range(1, episodes + 1):
            eps = self.eps_by_episode(episode)
            loss, wins, dones = self.train_episode(eps)

            passes = self.env.eval(self.model, debug = False)
            print(f"Episode: {episode:-2d} {'Yes!' if passes else 'Nope'}, Eps = {eps:.2f}, Total Wins: {wins:5g}, Total Dones: {dones:5g}")

        self.target_model.load_state_dict(self.model.state_dict())
