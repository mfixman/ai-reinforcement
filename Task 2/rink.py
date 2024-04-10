import torch

from torch import optim

from DQN import DQN
from SkatingRinkEnv import SkatingRinkEnv
from Trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = dict(
    hidden_size = 30,

    win_distance = 1,
    lose_distance = 7,
    max_eval_steps = 100,

    eps_start = 1,
    eps_end = .1,
    eps_decay = 400,

    batch_size = 1000,
    actions_size = 1000,
    buf_multiplier = 100,
    train_steps = 100,

    train_episodes = 1000,
    gamma = .9,
    eval_steps = 500,
)

def main():
    env = SkatingRinkEnv(config)
    model = DQN(env.state_n, config['hidden_size'], env.actions_n).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = .001)

    Trainer(config, env, model, optimizer).train()
    result = env.eval(model, debug = False)
    if result:
        print('Finished!')
    else:
        print('Not finished :-(')

if __name__ == '__main__':
    main()
