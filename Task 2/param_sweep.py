import argparse
import torch

from torch import optim

from argparse import ArgumentParser
from DQN import DQN
from SkatingRinkEnv import SkatingRinkEnv
from Trainer import Trainer

from itertools import product
import logging
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description = "Parameter sweep")

    parser.add_argument('--win_distance', type = float, default = 1., help = 'Winning distance.')
    parser.add_argument('--lose_distance', type = float, default = 10., help = 'Losing distance.')

    parser.add_argument('--methods', nargs = '+', type = str, default = [Trainer.DQN], choices = Trainer.Methods, help = 'Which learning methods to use.')
    parser.add_argument('--hidden_sizes', nargs = '+', type = int, default = [64], help = 'Size of the hidden layers.')
    parser.add_argument('--lrs', nargs = '+', type = float, default = [0.001], help = 'Learning rates.')
    parser.add_argument('--gammas', nargs = '+', type = float, default = [0.9], help = 'Discount factors for future rewards.')
    parser.add_argument('--eps_starts', nargs = '+', type = float, default = [1.0], help = 'Starting value of epsilon.')

    parser.add_argument('--max_eval_steps', type = int, default = 250, help = 'Maximum evaluation steps.')

    parser.add_argument('--eps_end', type = float, default = 0, help = 'Final value of epsilon.')
    parser.add_argument('--eps_decay', type = int, default = 500, help = 'Decay rate of epsilon.')

    parser.add_argument('--batch_size', type = int, default = 5000, help = 'Batch size for training.')
    parser.add_argument('--actions_size', type = int, default = 1000, help = 'Number of actions.')
    parser.add_argument('--buf_multiplier', type = int, default = 100, help = 'Buffer size multiplier.')
    parser.add_argument('--train_steps', type = int, default = 250, help = 'Number of training steps.')
    parser.add_argument('--train_episodes', type = int, default = 500, help = 'Number of training episodes.')
    parser.add_argument('--eval_steps', type = int, default = 250, help = 'Number of evaluation steps.')

    parser.add_argument('--max_rewards', type = int, default = 1000, help = 'Maximum reward.')

    parser.add_argument('--tau', type = float, default = 1.0, help = 'Target network soft update parameter tau.')
    parser.add_argument('--tau_decay', type = float, default = 1.0, help = 'Decay rate of tau.')
    parser.add_argument('--update_freq', type = int, default = 50, help = 'Update frequency for target network.')

    parser.add_argument('--output_file', help = 'Output file')

    return vars(parser.parse_args())

def main():
    config = parse_args()

    out = sys.stdout
    if config['output_file'] is not None:
        out = open(config['output_file'], 'w')

    logging.basicConfig(
        level = logging.INFO,
        format = '[%(asctime)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
    )

    print(f'method,hidden_size,lr,gamma,eps_start,win_rate,best_episode,best_dones,avg_reward,loss,q_step', file = out)

    combinations = product(config['methods'], config['hidden_sizes'], config['lrs'], config['gammas'], config['eps_starts'])
    for method, hidden_size, lr, gamma, eps_start in combinations:
        logging.info(f'Attempting {method} {hidden_size} {lr} {gamma} {eps_start}:')

        this_config = config.copy()
        this_config['method'] = method
        this_config['hidden_size'] = hidden_size
        this_config['lr'] = lr
        this_config['gamma'] = gamma
        this_config['eps_start'] = eps_start

        torch.manual_seed(42)

        env = SkatingRinkEnv(this_config)
        model = DQN(env.state_n, hidden_size, env.actions_n).to(device)
        model_target = DQN(env.state_n, hidden_size, env.actions_n).to(device)
        optimizer = optim.AdamW(model.parameters(), lr = lr)

        trainer = Trainer(this_config, env, model, model_target, optimizer)
        best_episode, best_dones, loss, q_step = trainer.train(all_debug = False)

        states = env.dropin(1000)
        states, actions, new_states, rewards, dones = env.eval(model, states).tensors()

        dones = dones[-1]
        rewards = rewards[-1]
        win_rate = dones.sum() / dones.shape[0]
        avg_reward = rewards.sum() / rewards.shape[0]
        
        print(f'{method},{hidden_size},{lr},{gamma},{eps_start},{win_rate:g},{best_episode},{best_dones},{avg_reward:g},{loss},{q_step}', file = out)

if __name__ == '__main__':
    main()
