import argparse
import torch

from torch import optim

from argparse import ArgumentParser
from DQN import DQN
from SkatingRinkEnv import SkatingRinkEnv
from Trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description = "Setup command line arguments for the model training configuration.")

    parser.add_argument('--hidden_size', type = int, default = 64, help = 'Size of the hidden layer.')
    parser.add_argument('--win_distance', type = float, default = 1., help = 'Winning distance.')
    parser.add_argument('--lose_distance', type = float, default = 7., help = 'Losing distance.')
    parser.add_argument('--max_eval_steps', type = int, default = 100, help = 'Maximum evaluation steps.')

    parser.add_argument('--eps_start', type = float, default = 1.0, help = 'Starting value of epsilon.')
    parser.add_argument('--eps_end', type = float, default = 0.001, help = 'Final value of epsilon.')
    parser.add_argument('--eps_decay', type = int, default = 200, help = 'Decay rate of epsilon.')

    parser.add_argument('--batch_size', type = int, default = 5000, help = 'Batch size for training.')
    parser.add_argument('--actions_size', type = int, default = 1000, help = 'Number of actions.')
    parser.add_argument('--buf_multiplier', type = int, default = 100, help = 'Buffer size multiplier.')
    parser.add_argument('--train_steps', type = int, default = 100, help = 'Number of training steps.')
    parser.add_argument('--train_episodes', type = int, default = 800, help = 'Number of training episodes.')
    parser.add_argument('--gamma', type = float, default = 0.9, help = 'Discount factor for future rewards.')
    parser.add_argument('--eval_steps', type = int, default = 500, help = 'Number of evaluation steps.')

    parser.add_argument('--max_rewards', type = int, default = 1000, help = 'Maximum reward.')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate.')

    parser.add_argument('--tau', type = float, default = 1.0, help = 'Target network soft update parameter tau.')
    parser.add_argument('--tau_decay', type = float, default = 1.0, help = 'Decay rate of tau.')
    parser.add_argument('--update_freq', type = int, default = 50, help = 'Update frequency for target network.')

    parser.add_argument('--method', type = str, default = Trainer.DQN, choices = Trainer.Methods, help = 'Which learning method to use.')

    parser.add_argument('--output_file', type = str, default = 'ice_skater.pth', help = 'Output file with the weights')

    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed')

    return vars(parser.parse_args())

def main():
    config = parse_args()
    torch.manual_seed(config['seed'])
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
    elif done and reward < 0:
        print('Finished and lost :-(((')
    else:
        print('Not finished :-(')

    torch.save({'weights': model.state_dict(), 'config': config}, config['output_file'])

if __name__ == '__main__':
    main()
