import gymnasium as gym
import numpy
import torch
import matplotlib.pyplot as plt

from ReplayBuffer import ReplayBuffer

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

    batch_size: int
    actions_size: int
    train_steps: int
    buf_multiplier: int

    def __init__(self, config : dict[str, Any], env: gym.Env, model: nn.Module, model_target: nn.Module, optimizer: optim.Adam):
        self.env = env
        self.model = model
        self.model_target = model_target
        self.model_target.load_state_dict(self.model.state_dict())
        self.config = config

        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()

        self.eps_start = self.config['eps_start']
        self.eps_end = self.config['eps_end']
        self.eps_decay = self.config['eps_decay']

        self.batch_size = self.config['batch_size']
        self.actions_size = self.config['actions_size']
        self.train_steps = self.config['train_steps']
        self.buf_multiplier = self.config['buf_multiplier']
        
        # 0 = DQN, 1 = DDQN
        self.method = self.config['method']
        
        self.max_rewards = self.config['max_rewards']
        self.train_episodes = self.config['train_episodes']
        self.q_log = []

    def eps_by_episode(self, episode: int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * numpy.exp(-1. * episode / self.eps_decay)

    @torch.no_grad()
    def choose_action(self, epsilon: float, states: tensor) -> tensor:
        size = (states.shape[0],)
        probs = tensor(numpy.random.choice([False, True], p = [epsilon, 1 - epsilon], size = size)).to(device)
        explorations = torch.randint(0, self.env.actions_n, size).to(device)
        explotations = self.model(states).argmax(1)
        return torch.where(probs, explotations, explorations)

    def train_episode(self, epsilon: float, method: int) -> tuple[float, int, int]:
        states = self.env.dropin(self.batch_size)
        q_step_log = torch.zeros(size=(self.train_steps,1))
        total_loss = tensor(.0)
        replays = ReplayBuffer(self.actions_size * self.buf_multiplier)

        total_wins = 0
        total_dones = 0
        for e in range(0, self.train_steps):
            actions = self.choose_action(epsilon, states)
            new_states, rewards, dones = self.env.steps(states, actions)

            total_wins += torch.sum(dones & (rewards > self.config['max_rewards']))
            total_dones += dones.sum()

            replays.add(states, actions, new_states, rewards, dones)

            states = new_states[~dones]
            loss, q_step = self.commit_gradient(*replays.sample(self.actions_size))
            total_loss += loss

            q_step_log[e] = torch.mean(q_step, dim=0)
            if total_dones == self.batch_size:
                break
        return total_loss, total_wins, total_dones, torch.mean(q_step_log, dim=0)

    def commit_gradient(self, states, actions, new_states, rewards, dones):
        gamma = self.config['gamma']
        
        
        self.optimizer.zero_grad()

        with torch.no_grad():
            
            # Get q values of the target model if DDQN is selected, else get q values of original model
            if(self.method == 1):
                next_q = self.model_target(new_states).max(axis = 1)[0]
            else:
                next_q = self.model(new_states).max(axis = 1)[0]
                
            expected_q = rewards + gamma * next_q * ~dones

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, expected_q)

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu(), expected_q
    
    def update_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def train(self):
        episodes = self.train_episodes
        for episode in range(1, episodes + 1):
            eps = self.eps_by_episode(episode)
            loss, wins, dones, q_step_log = self.train_episode(eps, self.config['method'])
            self.q_log.append(q_step_log)
            print(torch.mean(q_step_log, dim=0))
            reward, done = self.env.eval_single(self.model)
            print(f"Episode: {episode:-2d} {'Yes!' if done and reward > 0 else 'Nope' if done and reward <= 0 else 'Sad!'}, Epsilon = {eps:.2f}, Total Wins: {wins:5g}, Total Terminations: {dones:5g}, Loss : {loss:.4f}")
            self.plot()
            
    def plot(self):
        plt.figure(1)
        plt.clf()
        plt.plot(self.q_log)
        plt.pause(0.001)
        plt.title('Training')
        plt.xlabel('Episodes')
        plt.ylabel('Average Q')
        
        if(len(self.q_log)==self.train_episodes):
            plt.title('Average Q against Episodes')
            plt.show()