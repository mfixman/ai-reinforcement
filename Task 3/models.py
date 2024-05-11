import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Q_model(nn.Module):
    def __init__(self, obs_space, action_space, num_filters1, num_filters2, num_filters3, hidden_size1, kernel_size=3):
        super().__init__()
        self.input_shape = obs_space[0]
        self.conv1 = nn.Conv2d(self.input_shape, num_filters1, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(num_filters2, num_filters3, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*num_filters3, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, action_space)
        
    def forward(self, x):
        conv_out = F.relu(self.conv1(x))
        conv_out = F.relu(self.conv2(conv_out))
        conv_out = F.relu(self.conv3(conv_out))
        flatten = self.flatten(conv_out)
        dense_out = F.relu(self.fc1(flatten))
        output = self.fc2(dense_out)
        
        return output
