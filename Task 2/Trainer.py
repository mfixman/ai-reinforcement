import numpy
import torch

from SkatingRinkEnv import SkatingRinkEnv
from ReplayBuffer import ReplayBuffer

from torch import nn, optim, tensor, LongTensor, FloatTensor, BoolTensor
from typing import Any
from logger import Logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    DQN = 'DQN'
    TargetNetwork = 'TargetNetwork'
    DoubleDQN = 'DDQN'
    Methods = [DQN, TargetNetwork, DoubleDQN]

    env: SkatingRinkEnv
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

    def __init__(self, config : dict[str, Any], env: SkatingRinkEnv, model: nn.Module, model_target: nn.Module, optimizer: optim.Adam):
        if device == 'cpu':
            print('Warning! Using CPU')

        self.env = env
        self.model = model
        self.model_target = model_target
        self.model_target.load_state_dict(self.model.state_dict())
        
        # Target model shouldnt be updated
        self.model_target.eval()
        self.config = config

        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
        
        # self.logger = logger

        self.eps_start = self.config['eps_start']
        self.eps_end = self.config['eps_end']
        self.eps_decay = self.config['eps_decay']

        self.batch_size = self.config['batch_size']
        self.actions_size = self.config['actions_size']
        self.train_steps = self.config['train_steps']
        self.buf_multiplier = self.config['buf_multiplier']
        
        self.method = self.config['method']
        
        self.max_rewards = self.config['max_rewards']
        self.train_episodes = self.config['train_episodes']
        self.tau = self.config['tau']
        self.tau_decay = self.config['tau_decay']
        
        self.gamma = self.config['gamma']
        self.update_freq = self.config['update_freq']
        self.lr = self.config['lr']
        self.hidden_size = self.config['hidden_size']
        
        self.q_log = []

        self.reward_log = []

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
            average_q_per_agent = torch.mean(q_step_log, dim=0)
            
            
            # Update target model every m steps
            if e % self.config['update_freq'] == 0:
                self.update_target()
            
            # Decay Tau
            self.tau *= self.config['tau_decay']
            
            if total_dones == self.batch_size:
                break
            
            
        return total_loss, total_wins, total_dones, average_q_per_agent, 

    def commit_gradient(self, states, actions, new_states, rewards, dones):
        # Input variables:
        # states: current available states
        # actions: actions selected based on Boltzmann policy with respect to epsilon and model policy
        # new_states: new states obtained from selecting the action
        # rewards: rewards obtained from the state action pair
        # dones: binary value with dimensions [1, batch_size] depecting which run has finished in the batch
        
        gamma = self.config['gamma']
        self.optimizer.zero_grad()

        # with torch.no_grad(): # needs grad to back propagate
        # Get q values of the target model if Target Network/DDQN is selected, else get q values of original model

        if(self.method == Trainer.DQN):
            next_q = self.model(new_states).max(axis = 1)[0]
            expected_q = rewards + gamma * next_q * ~dones
        elif(self.method == Trainer.TargetNetwork):
            # Simply evaluate the action of the MAIN network using the TARGET network
            next_q = self.model_target(new_states).max(axis = 1)[0]
            expected_q = rewards + gamma * next_q * ~dones
        elif(self.method == Trainer.DoubleDQN):
            # For DDQN, MAIN network is used to select action
            next_model_action = self.model(new_states).max(axis = 1)[1]

            # TARGET network is used to evaluate the action of the MAIN network
            next_q =torch.gather(self.model_target(new_states), 1, next_model_action.unsqueeze(1))[~dones]
            expected_q = rewards + gamma

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, expected_q)

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu(), expected_q
    
    def update_target(self):
        model_state_dict = self.model.state_dict()
        target_state_dict = self.model_target.state_dict()
        for param in target_state_dict:
            target_state_dict[param] = model_state_dict[param] * self.tau + target_state_dict[param] * (1 - self.tau)
        self.model_target.load_state_dict(target_state_dict)
        self.model_target.eval()

    def train(self):
        episodes = self.train_episodes
        for episode in range(1, episodes + 1):
            eps = self.eps_by_episode(episode)
            loss, wins, dones, q_step_log = self.train_episode(eps, self.config['method'])
            self.q_log.append(q_step_log.detach().item())
            reward, done = self.env.eval_single(self.model)
            self.reward_log.append(reward)
            # ["method", "eps_decay", "tau_decay", "batch_size", "actions_size", "gamma", "update_freq", "lr", "hidden_size", "q_values", "reward"]
            # self.logger.log({"Q": q_step_log.detach().item(),
            #                  "Reward": reward})
            # print(f"Episode: {episode:-2d}\t{'Yes!' if done and reward > 0 else 'Nope' if done and reward <= 0 else 'Sad!'}\tEps = {eps:.2f}\tWins: {wins:5g}\tFinish: {dones:5g}\tLoss: {int(loss):-9d}")
            # self.plot()
            
    def plot(self):
        return

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
