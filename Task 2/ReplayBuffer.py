import numpy
import torch

from torch import tensor, FloatTensor, LongTensor, BoolTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    states: None | FloatTensor
    actions: None | LongTensor
    new_states: None | FloatTensor
    rewards: None | LongTensor
    dones: None | BoolTensor

    initialised: bool

    max_size: None | int

    def __init__(self, max_size = None):
        self.states = None
        self.actions = None
        self.new_states = None
        self.rewards = None
        self.dones = None
        self.initialised = False

        self.max_size = max_size

    def set(self, states, actions, new_states, rewards, dones):
        self.states = states.detach().clone()
        self.actions = actions.detach().clone()
        self.new_states = new_states.detach().clone()
        self.rewards = rewards.detach().clone()
        self.dones = dones.detach().clone()
        self.initialised = True

    def add_single(self, *args):
        self.add([x.unsqueeze(0) for x in args])

    def add(self, states, actions, new_states, rewards, dones):
        if not self.initialised:
            self.set(states, actions, new_states, rewards, dones)
            return

        selector = None
        if self.max_size is not None:
            selector = -self.max_size

        self.states = torch.cat([self.states, states])[selector:]
        self.actions = torch.cat([self.actions, actions])[selector:]
        self.new_states = torch.cat([self.new_states, new_states])[selector:]
        self.rewards = torch.cat([self.rewards, rewards])[selector:]
        self.dones = torch.cat([self.dones, dones])[selector:]

    def sample(self, amount: int) -> tuple[tensor, tensor, tensor, tensor, tensor]:
        if not self.initialised:
            raise ValueError('Uninitialised buffer!')

        n = self.states.shape[0]
        replays_idx = numpy.random.randint(0, n, size = amount)

        return (
            self.states[replays_idx],
            self.actions[replays_idx],
            self.new_states[replays_idx],
            self.rewards[replays_idx],
            self.dones[replays_idx]
        )

    def tensors(self) -> tuple[tensor, tensor, tensor, tensor, tensor]:
        if not self.initialised:
            raise ValueError('Uninitialised buffer!')

        return self.states, self.actions, self.new_states, self.rewards, self.dones

    def shape(self) -> tuple:
        smallest = min(len(x.shape) for x in self.tensors())
        shapes = [x.shape[:smallest] for x in self.tensors()]
        if len(set(shapes)) != 1:
            raise ValueError(f'Inconsistent shapes! {shapes}')

        return shapes[0]
