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

    end: ndarray

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

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.end = config['end']

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

        distances = (new_states[:, 0:2] - tensor(self.end).to(device)).square().sum(axis = 1).sqrt()

        in_win_range = distances < self.config['win_distance']
        out_play_range = distances >= self.config['lose_distance']

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

    def eval(self, model: nn.Module, debug = True) -> bool:
        state = self.zeros(1)
        with torch.no_grad():
            for e in range(1, 1 + self.config['max_eval_steps']):
                q_values = model(state)
                action = q_values.max(dim = 1)[1]

                new_state, _, done = self.steps(state, action)

                dist = (((state[0][0] - 2) ** 2 + (state[0][1] - 2) ** 2) ** (1/2))
                ang = (math.atan2(state[0][0] - 2, state[0][1] - 2) / numpy.pi)

                if debug:
                    print(f'{e:-2d}: [{state[0][0]: 3.3f} {state[0][1]: 3.3f} {state[0][2]: 3.3f}] [{dist: 3.3f} {ang: 3.3f}] -> {action.item()}')

                if done.all():
                    if debug:
                        print('Finished:-)')
                    return True

                state = new_state

        if debug:
            print('Never finished:-(')

        return False

    def reset(self) -> ndarray:
        self.state = numpy.zeros(3, dtype = numpy.float32)
        return self.state

    @classmethod
    def zeros(cls, batch_size: int) -> tensor:
        return torch.zeros((batch_size, len(cls.state_vars)), device = device)
