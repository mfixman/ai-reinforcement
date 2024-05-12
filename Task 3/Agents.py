import torch
import random
import math
from models import PPO
import numpy as np
from replay import ReplayBuffer
import torch.nn as nn
import matplotlib.pyplot as plt

class PPO_Agent:
    def __init__(self, env, config, device, show_result=False):
        self.filename_policy = 'ppo_policy.pth'
        self.filename_value = 'ppo_value.pth'
        self.config = config
        self.env = env
        observation_space, _ = self.env.reset()
        action_space = env.action_space.n
        # Initialize model
        self.show_result = show_result

        self.policy_net = PPO(input_dim=observation_space.shape[0], output_dim=action_space)
        # self.value_net = PPO(input_dim=observation_space.shape[0], output_dim=action_space)
        # self.policy_net = Q_model(obs_space=observation_space.shape, action_space=action_space, num_filters1=config['num_filters1'], num_filters2=config['num_filters2'], num_filters3=config['num_filters3'], hidden_size1=config['hidden_size'])
        # self.value_net = Q_model(obs_space=observation_space.shape, action_space=action_space, num_filters1=config['num_filters1'], num_filters2=config['num_filters2'], num_filters3=config['num_filters3'], hidden_size1=config['hidden_size'])
        
        try:
            self.policy_net.load_state_dict(torch.load(self.filename_policy))
            # self.value_net.load_state_dict(torch.load(self.filename_value))
            print('Checkpoint Loaded')
        except: 
            print('No checkpoint, starting from scratch')
            pass
        
        self.device = device
        self.policy_net = self.policy_net.to(device)
        # self.value_net = self.value_net.to(device)
        
        self.render = self.config['render']
        
        self.optim_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.config['lr'])
        # self.optim_value = torch.optim.Adam(self.value_net.parameters(), lr=self.config['lr'])
        
        self.loss_func = nn.SmoothL1Loss()
        # Parameters for updating
        # self.update_freq = config['update_freq']
        
        self.lamda = config['lamda']
        
        # Parameters for action selection
        # self.eps_start = config['eps_start']
        self.eps_end = config['eps_end']
        # self.eps_decay = config['eps_decay']
        
        self.gamma = config['gamma']
        
        # Memory / Replay Buffer
        self.replaybuffer = ReplayBuffer(max_len=config['capacity'])
        
        self.score_plot = []
        self.t_plot = []
        
    def select_action(self, state, steps_done):
        #
        action_softmax, vals = self.policy_net(state)
        action_category = torch.distributions.Categorical(action_softmax)
        action = action_category.sample()
        
        return action.item(), action_softmax[:, action.item()].item()
    
    def learn(self, frame_counts):
        batch_size = self.config['batch_size']
        if (len(self.replaybuffer.memory) < batch_size):
            return
        
        # for i in range(math.ceil(frame_counts/batch_size)):
        for i in range(1):
            # Sample Batch from list of transitions
            batch = self.replaybuffer.sample(batch_size)
            state_shape = batch.state[0].shape
            
            state = torch.cat(batch.state)
            action = torch.Tensor(batch.action)
            reward = torch.tensor(batch.reward)
            
            ns_pointers = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool().to(self.device)
            
            try:
                next_state = torch.cat([ns for ns in batch.next_state if ns is not None])
            except:
                next_state = torch.tensor([])
            done = torch.tensor(batch.done)
            
            next_state_q = torch.zeros(batch_size, device=self.device)
            # State with size of (batch size, 1, 84, 84)
            # state = state.view(batch_size, -1,state_shape[-2], state_shape[-1]).float().to(self.device)
            state = state.float().to(self.device)
            action = action.long().unsqueeze(1).to(self.device)
            reward = reward.float().to(self.device)
            next_state = next_state.float().to(self.device)
            done = done.float().to(self.device)
            
            action_probs, values = self.policy_net(state)
            next_action_probs, next_values = self.policy_net(next_state)
            # sa_values = self.policy_net(state).gather(1,action.squeeze(1))
            with torch.no_grad():
                next_state_q[ns_pointers] = next_values.squeeze()
                expected_q = reward + self.gamma * next_state_q * (1-done)
                advantages = expected_q - values.squeeze()
                advantages = advantages.unsqueeze(1)
                
            action_softmax = action_probs.gather(1, action)
            prev_action_softmax = action_softmax.clone().detach()
            
            action_ratio = action_softmax / prev_action_softmax
            temp1 = action_ratio * advantages
            temp2 = torch.clamp(action_ratio, 1-self.eps_end, 1+self.eps_end) * advantages
            policy_loss = torch.min(temp1, temp2).mean()
            
            val_loss = self.loss_func(values, expected_q.detach().unsqueeze(1))
            
            loss = policy_loss + self.lamda * val_loss
            # expected_sa_values = (1-done) * (reward + next_state_q * self.gamma) + (done*reward)

            # self.optim_value.zero_grad()
            self.optim_policy.zero_grad()
            loss.backward()
            # self.optim_value.step()
            self.optim_policy.step()
            
            # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
            
            
    def train(self):
        print(self.device)
        max_score = 0
        self.policy_net.train()
        # self.value_net.train()
        
        for epoch in range(self.config['max_epochs']):
            # Reset done parameter (truncated or terminated = done)
            done = False
            
            frame_counts = 0
            # Begin environment
            score = 0
            observation, _ = self.env.reset()
            observation = np.array(observation) / 255.0
            state = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            
            for t in range(self.config['max_steps']):
                with torch.no_grad():
                    action, action_val = self.select_action(state, frame_counts)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                next_observation = np.array(next_observation) / 255.0
                # Render env
                if self.render:
                    self.env.render()
                    
                done = terminated or truncated
                
                frame_counts += 1
                
                if terminated:
                    next_state = None
                else:
                    next_state = torch.from_numpy(next_observation).float().unsqueeze(0).to(self.device)
                
                
                # Push to memory
                self.replaybuffer.add_transition(state, action, reward, next_state, float(done))

                state = next_state
                # Keep Score
                score += reward
                
                self.learn(frame_counts)
                
                # End early if step is finished
                if(done):
                    
                    break
                
            self.t_plot.append(t)
            self.score_plot.append(score)
            # self.plot(self.t_plot, mode='Steps')
            self.plot(self.score_plot, mode='Scores')
            print(f'Done, Episode:{epoch}, score={score}')
            if score > max_score:
                max_score = score
                torch.save(self.policy_net.state_dict(), self.filename_policy)
                print('Checkpoint Saved')

    def plot(self, episode_durations, mode):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if self.show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title(f'Training {mode}...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.savefig(f"{mode}.png", bbox_inches='tight')
        # plt.pause(0.001)
            
            