import torch

from torch import optim

from DQN import DQN
from SkatingRinkEnv import SkatingRinkEnv
from Trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = dict(
    hidden_size = 64,

    win_distance = 1,
    lose_distance = 7,
    max_eval_steps = 100,

    eps_start = 1,
    eps_end = .1,
    eps_decay = 500,

    batch_size = 10,
    actions_size = 1000,
    buf_multiplier = 100,
    train_steps = 100,

    train_episodes = 800,
    gamma = .9,
    eval_steps = 500,
    
    max_rewards = 1000,
    lr = 0.001,
    update_freq = 50,
    
    # DQN = 0, DDQN = 1
    method = 1,
)

def main():
    env = SkatingRinkEnv(config)
    model = DQN(env.state_n, config['hidden_size'], env.actions_n).to(device)
    model_target = DQN(env.state_n, config['hidden_size'], env.actions_n).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = config['lr'])

    #Training Step
    Trainer(config, env, model, model_target, optimizer).train()
    
    #Evaluation Step
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
