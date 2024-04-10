import torch

from torch import optim

from DQN import DQN
from SkatingRinkEnv import SkatingRinkEnv
from Trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    env = SkatingRinkEnv()
    model = DQN(env.observation_space.shape[0], output_dim = env.action_space.n).to(device)
    target_model = DQN(env.observation_space.shape[0], output_dim = env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr = .001)

    Trainer(env, model, target_model, optimizer).train()
    result = env.eval(model, debug = False)
    if result:
        print('Finished!')
    else:
        print('Not finished :-(')

if __name__ == '__main__':
    main()
