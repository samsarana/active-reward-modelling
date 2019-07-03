"""Class for a Linear Quadratic Regulator.
"""
import math, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LQR(nn.Module):
    def __init__(self, obs_shape, act_shape, args):
        super().__init__()
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.Q = nn.Parameter(args.init_var * torch.randn(obs_shape, obs_shape))
        self.R = nn.Parameter(args.init_var * torch.randn(act_shape, act_shape))
        self.c = nn.Parameter(args.init_var * torch.randn(1)) # shape: torch.Size([1])

    def forward(self, obs, act):
        # TODO possibly I shouldn't torchify and reshape the tensors inside forward(.)
        obs = torch.tensor(obs).float().unsqueeze(-1) # shaped from (4,) to (4,1)
        act = torch.tensor(act).float().unsqueeze(-1).unsqueeze(-1) # shaped from () to (1,1)
        sTQs = torch.matmul( torch.matmul(obs.transpose(-1,-2), self.Q), obs)
        aTRa = torch.matmul( torch.matmul(act.transpose(-1,-2), self.R), act)
        return sTQs + aTRa + self.c


class ReplayBufferLQR():
    """NB two changes from DQN ReplayBuffer which affect downstream
       computation as well:
       (i) buffer takes only 3-tuples
       (ii) sample function returns np.array not torch.tensor
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward):
        """In CartPole env these are the types:
           state: np.array == Box(4)
           action: int == Discrete(2)
           reward: int in {1,-10}
           next_state: np.array == Box(4)
           done: bool
        """
        self.buffer.append((state, action, reward))
    
    def sample(self, batch_size):
        """Zip returns tensors for each unpacked element;
           np contructors are happy to take tuples of np.array / int / bool
        """
        state, action, reward = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state),
                np.array(action),
                np.array(reward))
    
    def __len__(self):
        return len(self.buffer)