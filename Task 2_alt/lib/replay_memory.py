from collections import namedtuple, deque
import random

# Function for Memory (referenced from torch library)
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
