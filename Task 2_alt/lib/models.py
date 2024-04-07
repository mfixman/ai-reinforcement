import torch
import torch.nn as nn
import torch.nn.functional as F

class myDQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden1=128, n_hidden2=128):
        super(myDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer3 = nn.Linear(n_hidden2, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x