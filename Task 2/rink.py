import argparse
import torch

from torch import optim
from DQN import DQN
from SkatingRinkEnv import SkatingRinkEnv
from Trainer import Trainer
from logger import Logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# filepath = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description = "Setup command line arguments for the model training configuration.")

    parser.add_argument('--hidden_size', type = int, default = 64, help = 'Size of the hidden layer.')
    parser.add_argument('--win_distance', type = int, default = 1, help = 'Winning distance.')
    parser.add_argument('--lose_distance', type = int, default = 7, help = 'Losing distance.')
    parser.add_argument('--max_eval_steps', type = int, default = 100, help = 'Maximum evaluation steps.')

    parser.add_argument('--eps_start', type = float, default = 1.0, help = 'Starting value of epsilon.')
    parser.add_argument('--eps_end', type = float, default = 0.1, help = 'Final value of epsilon.')
    parser.add_argument('--eps_decay', type = int, default = 500, help = 'Decay rate of epsilon.')

    parser.add_argument('--batch_size', type = int, default = 5000, help = 'Batch size for training.')
    parser.add_argument('--actions_size', type = int, default = 1000, help = 'Number of actions.')
    parser.add_argument('--buf_multiplier', type = int, default = 100, help = 'Buffer size multiplier.')
    parser.add_argument('--train_steps', type = int, default = 100, help = 'Number of training steps.')
    parser.add_argument('--train_episodes', type = int, default = 200, help = 'Number of training episodes.')
    parser.add_argument('--gamma', type = float, default = 0.9, help = 'Discount factor for future rewards.')
    parser.add_argument('--eval_steps', type = int, default = 500, help = 'Number of evaluation steps.')

    parser.add_argument('--max_rewards', type = int, default = 1000, help = 'Maximum reward.')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate.')

    parser.add_argument('--tau', type = float, default = 1.0, help = 'Target network soft update parameter tau.')
    parser.add_argument('--tau_decay', type = float, default = 1.0, help = 'Decay rate of tau.')
    parser.add_argument('--update_freq', type = int, default = 50, help = 'Update frequency for target network.')

    parser.add_argument('--method', type = str, default = Trainer.DQN, choices = Trainer.Methods, help = 'Which learning method to use.')

    return vars(parser.parse_args())

def main():
    config = parse_args()
    env = SkatingRinkEnv(config)
    model = DQN(env.state_n, config['hidden_size'], env.actions_n).to(device)
    model_target = DQN(env.state_n, config['hidden_size'], env.actions_n).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = config['lr'])
    for method_config in [Trainer.DQN, Trainer.TargetNetwork, Trainer.DoubleDQN]:
        config['method'] = method_config
        for eps_decay_config in [50,150]:
            config['eps_decay'] = eps_decay_config  
            for tau_decay_config in [0.5,1.0]:
                config['tau_decay'] = tau_decay_config
                for batch_size_config in [32,64]:
                    config['batch_size'] = batch_size_config
                    for action_size_config in [32,64]:
                        config['actions_size'] = action_size_config
                        for gamma_config in [0.5,1.0]:
                            config['gamma'] = gamma_config
                            for update_freq_config in [10, 50]:
                                config['update_freq'] = update_freq_config
                                for lr_config in [0.01, 0.0001]:
                                    config['lr'] = lr_config
                                    for hidden_size_config in [32,128,256]:
                                        wandb_logger = Logger(f'{method_config}_{eps_decay_config}_{tau_decay_config}_{batch_size_config}_{action_size_config}_{gamma_config}_{update_freq_config}_{lr_config}_{hidden_size_config}', project="inm7077 AI")
                                        logger = wandb_logger.get_logger()
                                        config['hidden_size'] = hidden_size_config
                                        
                                        #Training Step
                                        myTrainer = Trainer(config, env, model, model_target, optimizer, logger)
                                        myTrainer.train()
                                        
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