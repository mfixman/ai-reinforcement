import gymnasium as gym
import torch
from lib import myDQN, ReplayBuffer, select_action, optimize_model, plot_durations
from matplotlib import pyplot as plt
import torch.optim as optim
from collections import namedtuple
import yaml
import os
from itertools import count
import numpy as np
import torch.nn as nn

#lib files
env = gym.make('LunarLander-v2')
config_path='config.yaml'
truepath=os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(truepath,config_path), 'r') as file:
    settings = yaml.safe_load(file)

num_episodes=settings['num_episodes']
lr=settings['lr']
batch_size=settings['batch_size']
gamma=settings['gamma']
eps_start=settings['eps_start']
eps_end=settings['eps_end']
eps_decay=settings['eps_decay']
tau=settings['tau']
greedy_epsilon=True
boltzmann = False

max_epochs=500
observation, info = env.reset(seed=42)
# print(np.digitize(observation).shape)
# print(observation.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_observations = len(observation)
# print(env.action_space)
n_actions = env.action_space.n

# Define DQN networks
# print(n_observations)
# print(n_actions)
my_DQN = myDQN(n_observations, n_actions).to(device)
my_DQN_target = myDQN(n_observations, n_actions).to(device)

# Copy the original DQN for double DQN
my_DQN_target.load_state_dict(my_DQN.state_dict())

# Define optimizer
optimizer = optim.AdamW(my_DQN.parameters(), lr=lr, amsgrad=True)

# Define Replay Memory
transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
memory = ReplayBuffer(10000)

total_steps = 0
episode_list=[]
for epoch in range(num_episodes):
    state, info= env.reset(seed=42)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # print(epoch)
    for t_iter in count():
        action = select_action(state,env,my_DQN,total_steps, eps_end, eps_end, eps_decay, device)
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        total_steps += 1
        done= terminated or truncated
        # print(done)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
        memory.push(state, action, next_state, reward)
        
        state = next_state
        loss_func = nn.SmoothL1Loss()
        optimize_model(memory, transition, my_DQN, my_DQN_target, optimizer, gamma, batch_size, device, loss_func)
        
        target_policy_dict = my_DQN_target.state_dict()
        policy_policy_dict = my_DQN.state_dict()
        
        for key in policy_policy_dict:
            target_policy_dict[key] = policy_policy_dict[key] * tau + target_policy_dict[key] * (1 - tau)
        my_DQN_target.load_state_dict(target_policy_dict)

        if done:
            episode_list.append(t_iter + 1)
            plot_durations(episode_list)
            break
     
    plot_durations(episode_list, show_result=True)
    plt.ioff()
    plt.show(block=False)
        
env.close()