import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch
import numpy as np
from utils import Atari_Prep
from Agents import DQN_Agent


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

config = dict(
    #Environment Variables
    seed=42,
    env_name = 'BreakoutDeterministic-v4',
    width = 84,
    height = 84,
    render = False,
    
    # DQN variables
    hidden_size = 512,
    num_filters1 = 32,
    num_filters2 = 64,
    num_filters3 = 64,
    
    # Training variables
    lr = int(2.5e-3),
    gamma = 0.999,
    epsilon = 1.0,
    eps_start = 1.0,
    eps_end = 0.1,
    eps_decay = 100,
    batch_size = 4,
    max_steps = 75000000,
    max_epochs = 200,
    
    update_freq = 20,  #Update frequency for Target Network
    
    # Plotting variables
    plot_epsilon_random = 69000,
    plot_epsilon_greedy = 1000000,
    
    # Memory variables
    capacity = int(4e6),
    
    
    
)

# Create Gym Environment
if(config['render']):
    env = gym.make(config['env_name'], render_mode='human')
else:
    env = gym.make(config['env_name'])

env.seed(config['seed'])
# Preprocess gym environment
actions = env.action_space.n
input_size = (actions, config['width'], config['height'])

env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
env = gym.wrappers.FrameStack(env, 4)
# print(env.observation_space)



DQN_Agent = DQN_Agent(env, config, device)
DQN_Agent.train()



