"""Classes and functions to do Q-learning"""

import gym, random, time
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, args):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, args.h1_agent),
            nn.ReLU(),
            nn.Linear(args.h1_agent, args.h2_agent),
            nn.ReLU(),
            nn.Linear(args.h2_agent, num_actions)
        )
        self.num_actions = num_actions
        self.batch_size = args.batch_size_agent
        self.gamma = args.gamma
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay # exponential decay of epsilon after learning step
        self.epsilon_stop = args.epsilon_stop
        self.tau = args.target_update_tau

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon=0):
        """For Q-networks, computing action and forward pass are NOT
           the same thing! (unlike for policy networks)
           Takes Box(4) and returns Discrete(2)
           Box(4) = R^4, represented as np.ndarray, shape=(4,), dtype=float32
           Discrete(2) = {0, 1} where 0 and 1 are standard integers
           [NB previously in my Gridworld: took a tuple (int, int) and returns scalar tensor with dtype torch.long]
        """
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float)
            q_value = self(state)
        # max is along non-batch dimension, which may be 0 or 1 depending on whether input is batched (action selection: not batched; computing loss: batched)
            _, action_tensor  = q_value.max(-1) # max returns a (named)tuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
            action = int(action_tensor)
        else:
            action = random.randrange(0, self.num_actions)
        return action


class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """In CartPole env these are the types:
           state: np.array == Box(4)
           action: int == Discrete(2)
           reward: int in {1,-10}
           next_state: np.array == Box(4)
           done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Zip returns tensors for each unpacked element;
           torch contructors are happy to take tuples of np.array / int / bool
        """
        # TODO is next line a bottleneck? if so, can speed up sampling
        # by implementing buffer with numpy ringbuffer?
        # from numpy_ringbuffer import RingBuffer
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size)) # samples without replacement
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)


def q_learning_loss(q_net, q_target, replay_buffer, args,
                    mean_rew=None, var_rew=None, reward_model=None):
    """Defines the loss function above
       Help on interpreting variables:
       Each dimension of the batch pertains to one transition, i.e. one 5-tuple
            (state, action, reward, next_state, done)
       q_values : batch_dim x n_actions
       q_value : batch_dim (x 1 -> squeezed). tensor of Q-values of action taken
                  in each transition (transitions are sampled from replay buffer)
       next_q_values : batch_dim x n_actions (detached)
       next_q_value : as above, except has Q-values of optimal next action after
                       each transition, rather than action taken by agent
       expected_q_value : batch_dim. implements y_i.
    """
    state, action, reward, next_state, done = replay_buffer.sample(q_net.batch_size)
    # compute r_hats according to current reward_model and/or normalise rewards
    normalise_rewards = True if mean_rew and var_rew else False
    if normalise_rewards:
        if reward_model:
            sa_pair = torch.cat((state, action.unsqueeze(1).float()), dim=1)
            assert isinstance(reward_model, nn.Module)
            reward_model.eval() # turn off dropout at 'test' time i.e. when getting rewards to send to DQN
            if args.no_ensemble_for_reward_pred:
                r_hat = reward_model.forward_single(sa_pair)
            else:
                r_hat = reward_model(sa_pair)
            norm_reward = (r_hat - mean_rew) / np.sqrt(var_rew + 1e-8)
        else:
            norm_reward = (reward - mean_rew) / np.sqrt(var_rew + 1e-8)
    else:
        norm_reward = reward

    q_values         = q_net(state)
    next_q_values    = q_target(next_state).detach() # params from target network held fixed when optimizing loss func

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # see https://colab.research.google.com/drive/1-6aNmf16JcytKw3BJ2UfGq5zkik1QLFm or https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    next_q_value, _  = next_q_values.max(-1) # max returns a (named)tuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
    expected_q_value = norm_reward + q_net.gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value).pow(2).mean() # mean is across batch dimension
    return loss


def init_agent(args):
    """Intitialises and returns the necessary objects for
       Deep Q-learning:
       Q-network, target network, replay buffer and optimizer.
    """
    q_net = DQN(args.obs_shape, args.n_actions, args)
    q_target = DQN(args.obs_shape, args.n_actions, args)
    q_target.load_state_dict(q_net.state_dict()) # set params of q_target to be the same
    replay_buffer = ReplayBuffer(args.replay_buffer_size) # TODO should I use old replay_buffer with reinitialised agent?
    optimizer_agent = optim.Adam(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent)
    return q_net, q_target, replay_buffer, optimizer_agent