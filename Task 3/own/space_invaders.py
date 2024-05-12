import gymnasium as gym
import torch
from torch import optim, nn

import matplotlib
from matplotlib import pyplot

device = 'cuda'

def main():
    torch.manual_seed(0)
    env = gym.make('ALE/SpaceInvaders-v5')

if __name__ == '__main__':
    main()
