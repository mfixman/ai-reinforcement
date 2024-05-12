import random
from collections import namedtuple, deque


# Replay Buffer, referenced from INM707 Labs
class ReplayBuffer(object):
    def __init__(self, max_len):
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        self.memory = deque([], maxlen=max_len)
        
    def add_transition(self, *args):
        self.memory.append(self.transition(*args))
        
    def sample(self, batch_size):
        curr_batch = random.sample(self.memory, batch_size)
        return self.transition(*zip(*curr_batch))
    
    def __len__(self):
        return len(self.memory)