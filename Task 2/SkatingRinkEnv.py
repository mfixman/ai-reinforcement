import math
import numpy
import torch

from numpy import ndarray, array
from torch import nn, tensor

from ReplayBuffer import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkatingRinkEnv:
    speed = .25
    ang_acc = 1/20 * (2 * numpy.pi)

    actions_n = 3
    actions = {
        'left': 0,
        'straight': 1,
        'right': 2,
    }

    state_n = 4
    state_vars = {
        'y': 0,
        'x': 1,
        'phi': 2,
        'theta': 3,
    }

    win_distance: int
    lose_distance: int
    max_eval_steps: int

    def __init__(self, config, device = device):
        self.win_distance = config['win_distance']
        self.lose_distance = config['lose_distance']
        self.max_eval_steps = config['max_eval_steps']
        self.device = device

    @staticmethod
    def torch_choice(*args : tuple[tensor, tensor], else_action: tensor) -> tensor:
        result = else_action.clone()
        for cond, action in args[::-1]:
            result = torch.where(cond, action, result)

        return result

    @staticmethod
    def clip_angle(phis):
        return phis.remainder(tensor(2 * torch.pi))

    def rewards_dones(self, states : tensor) -> tuple[tensor, tensor]:
        distances = states[:, 0:2].square().sum(axis = 1).sqrt()
        in_win_range = distances < self.win_distance
        out_play_range = distances >= self.lose_distance

        rewards = self.torch_choice(
            (in_win_range, 10000),
            (out_play_range, -10000),
            else_action = -distances,
        )

        dones = self.torch_choice(
            (in_win_range, True),
            (out_play_range, True),
            else_action = tensor(False).to(self.device),
        )

        return rewards, dones

    def steps(self, states: tensor, actions: tensor) -> tuple[tensor, tensor, tensor]:
        signs = actions - 1

        ys, xs, phis, thetas = states.T
        thetas_d = thetas + signs * self.ang_acc
        phis_d = self.clip_angle(phis + thetas_d)
        new_states = torch.stack(
            [
                ys + self.speed * torch.sin(phis_d),
                xs + self.speed * torch.cos(phis_d),
                phis_d,
                thetas_d,
            ]
        ).T

        _, finished = self.rewards_dones(states)
        rewards, dones = self.rewards_dones(new_states)

        return (
            torch.where(finished.unsqueeze(1), states, new_states),
            torch.where(finished, 0, rewards),
            torch.where(finished, True, dones),
        )

    @torch.no_grad()
    def eval(self, model : nn.Module, states : tensor) -> ReplayBuffer:
        ret = ReplayBuffer(100000)
        for e in range(1, 1 + self.max_eval_steps):
            q_values = model(states)
            actions = q_values.max(dim = 1)[1]

            new_states, rewards, dones = self.steps(states, actions)
            if e == self.max_eval_steps:
                rewards = torch.where(~dones, -10000, rewards)

            ret.add_single(states, actions, new_states, rewards, dones)

            states = new_states
        return ret

    @torch.no_grad()
    def eval_single(self, model : nn.Module) -> tuple[int, bool]:
        return self.eval_many(model, 1)

    @torch.no_grad()
    def eval_many(self, model: nn.Module, batch_size: int) -> tuple[float, int]:
        guys = self.dropin(batch_size)
        states, actions, new_states, rewards, dones = self.eval(model, guys).tensors()
        return rewards[-1].mean(dtype = torch.float32), dones[-1].sum()

    @classmethod
    def zeros(cls, batch_size: int) -> tensor:
        return torch.zeros((batch_size, cls.state_n)).to(self.device)

    def lose_distance_at(self, step):
        return (self.lose_distance - self.win_distance) / step + self.win_distance

    def dropin(self, batch_size: int, max_dist = None) -> tensor:
        min_dist = 1/3 * (self.lose_distance - self.win_distance) + self.win_distance

        if max_dist is None:
            max_dist = 5/6 * (self.lose_distance - self.win_distance) + self.win_distance

        p = torch.rand(batch_size) * (max_dist - min_dist) + min_dist
        alpha = torch.rand(batch_size) * (2 * numpy.pi)

        ys = p * torch.sin(alpha)
        xs = p * torch.cos(alpha)
        phis = torch.rand(batch_size) * (2 * numpy.pi)
        thetas = torch.zeros(batch_size)
        return torch.stack([ys, xs, phis, thetas]).T.to(self.device)
