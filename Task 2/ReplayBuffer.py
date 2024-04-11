import numpy
import torch

from torch import tensor, FloatTensor, LongTensor, BoolTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    states: list[FloatTensor]
    actions: list[LongTensor]
    new_states: list[FloatTensor]
    rewards: list[LongTensor]
    dones: list[BoolTensor]

    max_size: None | int

    def __init__(self, max_size = None):
        self.states = []
        self.actions = []
        self.new_states = []
        self.rewards = []
        self.dones = []

        self.max_size = max_size

    def add_single(self, *args):
        self.add(*[x.unsqueeze(0) for x in args])

    def add(self, states, actions, new_states, rewards, dones):
        self.states.append(states)
        self.actions.append(actions)
        self.new_states.append(new_states)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def coalesce(self):
        selector = None
        if self.max_size is not None:
            selector = -self.max_size

        self.states = [torch.cat(self.states)[selector:]]
        self.actions = [torch.cat(self.actions)[selector:]]
        self.new_states = [torch.cat(self.new_states)[selector:]]
        self.rewards = [torch.cat(self.rewards)[selector:]]
        self.dones = [torch.cat(self.dones)[selector:]]

    def sample(self, amount: int) -> tuple[tensor, tensor, tensor, tensor, tensor]:
        self.coalesce()
        replays_idx = numpy.random.randint(0, self.states[0].shape[0], size = amount)
        return tuple(x[replays_idx] for x in self.tensors())

    def tensors(self) -> tuple[tensor, tensor, tensor, tensor, tensor]:
        self.coalesce()
        return self.states[0], self.actions[0], self.new_states[0], self.rewards[0], self.dones[0]

    def shape(self) -> tuple:
        smallest = min(len(x.shape) for x in self.tensors())
        shapes = [x.shape[:smallest] for x in self.tensors()]
        if len(set(shapes)) != 1:
            raise ValueError(f'Inconsistent shapes! {shapes}')

        return shapes[0]
