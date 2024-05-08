import matplotlib
import seaborn
import sys
import torch

from torch import tensor
from matplotlib import pyplot

from DQN import DQN
from ReplayBuffer import ReplayBuffer
from SkatingRinkEnv import SkatingRinkEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_env_model(file):
    data = torch.load(file)
    config = data['config']

    env = SkatingRinkEnv(config)
    model = DQN(env.state_n, config['hidden_size'], env.actions_n).to(device)
    model.load_state_dict(data['weights'])

    return env, model

def get_results(env, model):
    states = env.dropin(4)
    return env.eval(model, states)

def plot(env, *tensors):
    seaborn.set()
    pyplot.ion()

    pyplot.gca().set_aspect('equal')
    pyplot.gca().add_patch(matplotlib.patches.Circle((0, 0), env.win_distance, fill = True))
    pyplot.gca().add_patch(matplotlib.patches.Circle((0, 0), env.lose_distance, fill = False))

    colors = seaborn.color_palette('husl', tensors[0].shape[1])
    for states, actions, new_states, rewards, dones, color in zip(*[x.unbind(1) for x in tensors], colors):
        ys, xs, phis = states[~dones].T.detach().cpu().numpy()
        yd, xd, phid = new_states[~dones].T.detach().cpu().numpy()
        pyplot.plot(xs + xd[[-1]], ys + yd[[-1]], color = color)

        if dones.any():
            y0, x0, phi0 = states[dones][0].detach().cpu().numpy()
            yf, xf, phif = new_states[dones][0].detach().cpu().numpy()
            pyplot.plot([x0, xf], [xf, yf], color = color)

    pyplot.show()
    input('Press any key to continue...')

def main():
    file = 'ice_skater.pth'
    if len(sys.argv) > 1:
        file = sys.argv[1]

    env, model = load_env_model(file)
    results = get_results(env, model)
    plot(env, *results.tensors())

if __name__ == '__main__':
    main()
