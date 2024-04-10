import math
import numpy
import torch

from gymnasium import spaces, Env
from numpy import ndarray, array
from torch import nn, tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkatingRinkEnv(Env):
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

    def steps(self, states: tensor, actions: tensor) -> tensor:
        signs = actions - 1

        ys, xs, phis = states.T
        d_phis = self.ang_speed * signs
        new_states = torch.stack(
            [
                ys + self.speed * torch.sin(phis + d_phis),
                xs + self.speed * torch.cos(phis + d_phis),
                phis + d_phis,
            ]
        ).T

        distances = new_states[:, 0:2].square().sum(axis = 1).sqrt()

        in_win_range = distances < self.win_distance
        out_play_range = distances >= self.lose_distance

        rewards = self.torch_choice(
            (in_win_range, 10000.),
            (out_play_range, -10000.),
            else_action = tensor(-1.).to(device),
        )

        dones = self.torch_choice(
            (in_win_range, True),
            (out_play_range, True),
            else_action = tensor(False).to(device),
        )

        return new_states, rewards, dones

    @torch.no_grad()
    def eval(self, model: nn.Module, state = None, debug = False) -> bool:
        if state is None:
            state = self.dropin(1)

        for e in range(1, 1 + self.max_eval_steps):
            q_values = model(state)
            action = q_values.max(dim = 1)[1]

            if debug:
                dist = (state[0][0] ** 2 + state[0][1] ** 2) ** (1/2)
                ang = math.atan2(state[0][0], state[0][1]) / numpy.pi
                print(f'{e:-2d}: [{state[0][0]: 3.3f} {state[0][1]: 3.3f} {state[0][2] / numpy.pi: 3.3f}] [{dist: 3.3f} {ang: 3.3f}] -> {action.item()}')

            state, reward, done = self.steps(state, action)
            if done.all():
                if debug:
                    dist = (state[0][0] ** 2 + state[0][1] ** 2) ** (1/2)
                    ang = math.atan2(state[0][0], state[0][1]) / numpy.pi
                    print(f' F: [{state[0][0]: 3.3f} {state[0][1]: 3.3f} {state[0][2] / numpy.pi: 3.3f}] [{dist: 3.3f} {ang: 3.3f}] -> {action.item()}')
                    if reward > 0:
                        print('Win :-)')
                    else:
                        print('Lose :-(')

                return True

        if debug:
            print('Never finished:-(')

        return False

    @classmethod
    def zeros(cls, batch_size: int) -> tensor:
        return torch.zeros((batch_size, cls.state_n)).to(device)

    def dropin(self, batch_size: int) -> tensor:
        dist = self.lose_distance / 2

        ys = torch.rand(batch_size) * (2 * dist) - dist
        xs = torch.rand(batch_size) * (2 * dist) - dist
        phis = torch.rand(batch_size) * (2 * numpy.pi) - numpy.pi
        return torch.stack([ys, xs, phis]).T.to(device)
