import math
import numpy
import torch

from numpy import ndarray, array
from torch import nn, tensor

from ReplayBuffer import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkatingRinkEnv:
    speed = .25
    ang_speed = 1/10 * (2 * numpy.pi)

    actions_n = 3
    actions = {
        'left': 0,
        'straight': 1,
        'right': 2,
    }

    state_n = 3
    state_vars = {
        'y': 0,
        'x': 1,
        'phi': 2,
    }

    win_distance: int
    lose_distance: int
    max_eval_steps: int

    def __init__(self, config):
        self.win_distance = config['win_distance']
        self.lose_distance = config['lose_distance']
        self.max_eval_steps = config['max_eval_steps']

    @staticmethod
    def torch_choice(*args : tuple[tensor, tensor], else_action: tensor) -> tensor:
        result = else_action.clone()
        for cond, action in args[::-1]:
            result = torch.where(cond, action, result)

        return result

    @staticmethod
    def clip_angle(phis):
        return phis + (phis.abs() > numpy.pi) * phis.sign() * -numpy.pi

    def rewards_dones(self, states : tensor) -> tuple[tensor, tensor]:
        distances = states[:, 0:2].square().sum(axis = 1).sqrt()
        in_win_range = distances < self.win_distance
        out_play_range = distances >= self.lose_distance

        rewards = self.torch_choice(
            (in_win_range, 10000),
            (out_play_range, -10000),
            else_action = tensor(-1).to(device),
        )

        dones = self.torch_choice(
            (in_win_range, True),
            (out_play_range, True),
            else_action = tensor(False).to(device),
        )

        return rewards, dones

    def steps(self, states: tensor, actions: tensor) -> tuple[tensor, tensor, tensor]:
        signs = actions - 1

        ys, xs, phis = states.T
        d_phis = self.ang_speed * signs
        new_states = torch.stack(
            [
                ys + self.speed * torch.sin(phis + d_phis),
                xs + self.speed * torch.cos(phis + d_phis),
                self.clip_angle(phis + d_phis),
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
        ret = ReplayBuffer()
        for e in range(1, 1 + self.max_eval_steps):
            q_values = model(states)
            actions = q_values.max(dim = 1)[1]

            new_states, rewards, dones = self.steps(states, actions)
            ret.add_single(states, actions, new_states, rewards, dones)

            states = new_states
        return ret

    @torch.no_grad()
    def eval_single(self, model : nn.Module) -> tuple[int, bool]:
        states, actions, new_states, rewards, dones = self.eval(model, self.dropin(1)).tensors()
        return rewards.sum().detach().item(), dones.any().detach().item()

    @classmethod
    def zeros(cls, batch_size: int) -> tensor:
        return torch.zeros((batch_size, cls.state_n)).to(device)

    def dropin(self, batch_size: int) -> tensor:
        min_dist = self.win_distance
        max_dist = self.lose_distance / 2

        p = torch.rand(batch_size) * (max_dist - min_dist) + min_dist
        alpha = torch.rand(batch_size) * (2 * numpy.pi) - numpy.pi

        ys = p * torch.sin(alpha)
        xs = p * torch.cos(alpha)
        phis = torch.rand(batch_size) * (2 * numpy.pi) - numpy.pi
        return torch.stack([ys, xs, phis]).T.to(device)
