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
    eps_decay = 200,

    batch_size = 4000,
    actions_size = 1000,
    buf_multiplier = 100,
    train_steps = 100,

    train_episodes = 250,
    gamma = .9,
    eval_steps = 500,
)

def main():
    env = SkatingRinkEnv(config)
    model = DQN(env.state_n, config['hidden_size'], env.actions_n).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = .001)

    Trainer(config, env, model, optimizer).train()
    reward, done = env.eval_single(model)
    if done and reward > 0:
        print('Finished and won :-)')
        torch.save({'weights': model.state_dict(), 'config': config}, 'ice_skater.pth')
    elif done and reward < 0:
        print('Finished and lost :-(((')
    else:
        print('Not finished :-(')

if __name__ == '__main__':
    main()
