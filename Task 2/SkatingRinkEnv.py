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

    end = array([2, 2])

    state: ndarray

    action_space = spaces.Discrete(3)
    observation_space: spaces.Box

    actions = {
        'left': 0,
        'straight': 1,
        'right': 2,
    }

    state_vars = {
        'y': 0,
        'x': 1,
        'phi': 2,
    }

    def __init__(self):
        super().__init__()

        space_range = (-100, 100)
        angle_range = (-numpy.pi, numpy.pi)

        ranges = array([space_range, space_range, angle_range])
        self.observation_space = spaces.Box(low = ranges[:, 0], high = ranges[:, 1])

        self.state = numpy.zeros(3, dtype = numpy.float32)

    @classmethod
    def steps(cls, states: tensor, actions: tensor) -> tensor:
        signs = actions - 1

        ys, xs, phis = states.T
        d_phis = cls.ang_speed * signs
        ds = torch.stack(
            [
                cls.speed * torch.sin(phis + d_phis),
                cls.speed * torch.cos(phis + d_phis),
                d_phis,
            ]
        ).T

        new_states = states + ds
        distances = (new_states[:, 0:2] - tensor(cls.end, device = device)).square().sum(axis = 1).sqrt()

        rewards = torch.where(distances < 1, 10000, -10)
        rewards = torch.where(distances >= 5, -10000, rewards)

        dones = torch.where((distances < 1) | (distances >= 5), True, False)

        return new_states, rewards, dones


    def step(self, action: int) -> tuple[ndarray, float, bool, dict[None, None]]:
        sign = action - 1

        y, x, phi = self.state

        self.state = array(
            [
                y + self.speed * numpy.sin(phi),
                x + self.speed * numpy.cos(phi),
                phi + self.ang_speed * sign,
            ],
            dtype = numpy.float32
        )

        distance = numpy.sqrt(numpy.sum((array([y, x]) - self.end) ** 2))

        if distance < 1:
            reward = 1000.
            done = True
        elif distance >= 100:
            # Out of bounds
            reward = -1000.
            done = True
        else:
            reward = -1.
            done = False

        return self.state, reward, done, {}

    def eval(self, model: nn.Module, debug = True) -> bool:
        self.reset()
        with torch.no_grad():
            for e in range(1, 101):
                q_values = model(tensor(self.state, device = device).unsqueeze(0).to(device)).squeeze().cpu().numpy()
                action = q_values.argmax()
                _, _, done, _ = self.step(action)

                dist = ((self.state[0] - 2) ** 2 + (self.state[1] - 2) ** 2) ** (1/2)
                ang = math.atan2(self.state[0] - 2, self.state[1] - 2) / numpy.pi

                if debug:
                    print(f'{e:-2d}: [{self.state[0]: 3.3f} {self.state[1]: 3.3f} {self.state[2]: 3.3f}] [{dist: 3.3f} {ang: 3.3f}] -> {action}')

                if done:
                    if debug:
                        print('Finished:-)')
                    return True

        if debug:
            print('Never finished:-(')

        return False

    def reset(self) -> ndarray:
        self.state = numpy.zeros(3, dtype = numpy.float32)
        return self.state

    @classmethod
    def zeros(cls, batch_size: int) -> tensor:
        return torch.zeros((batch_size, len(cls.state_vars)), device = device)
