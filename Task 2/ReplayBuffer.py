import numpy
import torch

from torch import tensor, FloatTensor, LongTensor, BoolTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    states: tensor
    actions: tensor
    new_states: tensor
    rewards: tensor
    dones: tensor

    max_size: int

    def __init__(self, max_size):
        self.states = FloatTensor().to(device)
        self.actions = LongTensor().to(device)
        self.new_states = FloatTensor().to(device)
        self.rewards = FloatTensor().to(device)
        self.dones = BoolTensor().to(device)

        self.max_size = max_size

    def add(self, states, actions, new_states, rewards, dones):
        self.states = torch.cat([self.states, states])[-self.max_size:]
        self.actions = torch.cat([self.actions, actions])[-self.max_size:]
        self.new_states = torch.cat([self.new_states, new_states])[-self.max_size:]
        self.rewards = torch.cat([self.rewards, rewards])[-self.max_size:]
        self.dones = torch.cat([self.dones, dones])[-self.max_size:]

    def sample(self, amount: int) -> tuple[tensor, tensor, tensor, tensor, tensor]:
        n = self.states.shape[0]
        replays_idx = numpy.random.randint(0, n, size = amount)

        return (
            self.states[replays_idx],
            self.actions[replays_idx],
            self.new_states[replays_idx],
            self.rewards[replays_idx],
            self.dones[replays_idx]
        )
