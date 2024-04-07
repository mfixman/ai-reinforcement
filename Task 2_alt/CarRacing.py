import gymnasium as gym
import torch
from lib import myDQN, ReplayMemory, select_action, optimize_model, plot_durations
from matplotlib import pyplot as plt
import torch.optim as optim
from collections import namedtuple
import yaml
import os
import numpy as np

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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

max_epochs=10000
terminated=0
truncated=0
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
memory = ReplayMemory(10000, transition=transition)

total_steps = 0
episode_list=[]

for epoch in range(num_episodes):
    state, info= env.reset(seed=42)
    print(state)
    state = np.reshape(state, (-1,3))
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while(True):
        action = select_action(state,env,my_DQN,total_steps, eps_end, eps_end, eps_decay, device)
        observation, reward, terminated, truncated, info = env.step(action.item())
        
        total_steps += 1
        done= terminated or truncated
        
        if terminated or truncated:
            done=True
            if(terminated):
                print('Environment is terminated')
                next_state = None
                # target = 
            else:
                print('Environment is truncated')
            
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
        memory.push(state, action, next_state, reward)
        
        state = next_state
        
        optimize_model(memory, transition, my_DQN, my_DQN_target, optimizer, gamma, batch_size, device)
        
        target_policy = my_DQN_target.state_dict()
        policy_policy = my_DQN.state_dict()
        
        for key in policy_policy:
            target_policy[key] = policy_policy[key] * tau + target_policy[key] * (1 - tau)
        target_policy.load_state_dict(target_policy)

        if done:
            episode_list.append(t + 1)
            plot_durations(episode_list)
            break
     
    plot_durations(episode_list, show_result=True)
    plt.ioff()
    plt.show()
        
env.close()