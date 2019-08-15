"""Classes and functions to do Q-learning"""

import gym, random, time, logging
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from reward_learning import RewardModelEnsemble

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

# TODO understand convolutions properly, and where the stack dimension should be
# Then go through rest of code and see what will break when obs.shape == (84,84,4)
# some things clearly will e.g. AgentExperience
# reward model (hmm, Christiano r^ : S x A -> R but it's a convnet...? now i can see
# why it made sense for Ibarz to parameterise as r^ : S -> R instead...)
# use the debugger to check the dims of everything

class CnnDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, args):
        super().__init__()
        self.num_actions = num_actions
        self.batch_size = args.batch_size_agent
        self.gamma = args.gamma
        self.tau = args.target_update_tau
        self.obs_shape = args.obs_shape_all
        self.convolutions = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # num_inputs == env.observation_space.shape[0] == (84,84,4)[0]. Still not sure this is going to work -- can other 2 input dims be left implicitly wtih Conv2d layers? Maybe need to use Conv3d..?
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            # nn.Linear(self.conv_layers_out_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    # def conv_layers_out_size(self):
    #     """Hack to compute size of 1st dim of
    #        output from features (0th dim
    #        is batch dim)
    #        Q: is this necessary b/c otherwise
    #        we'd have to redo annoying padding
    #        calculations every time we wanted
    #        to use the network with a different
    #        sized input?
    #     """
    #     print(self.obs_shape)
    #     print('convolutions_out_shape:', self.convolutions(torch.zeros(1, *self.obs_shape)).view(1, -1).shape)
    #     return self.convolutions(torch.zeros(1, *self.obs_shape)).view(1, -1).shape[1]
    
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
            state = state.unsqueeze(0)
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


def q_learning_loss(q_net, q_target, replay_buffer, args, reward_model=None,
                    normalise_rewards=True, true_reward_stats=None):
    """Defines the Q-Learning loss function.
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
    # import ipdb; ipdb.set_trace()
    state, action, true_reward, next_state, done = replay_buffer.sample(q_net.batch_size)
    # compute r_hats according to current reward_model and/or normalise rewards
    if reward_model: # RL from preferences
        sa_pair = torch.cat((state, action.unsqueeze(1).float()), dim=1)
        assert isinstance(reward_model, nn.Module)
        reward_model.eval() # turn off dropout at 'test' time i.e. when getting rewards to send to DQN
        
        if args.no_ensemble_for_reward_pred:
            assert isinstance(reward_model, RewardModelEnsemble)
            rew = reward_model.forward_single(sa_pair, normalise=normalise_rewards).detach()
        else:
            rew = reward_model(sa_pair, normalise=normalise_rewards).detach()
        rew = rew.squeeze()
        # import ipdb
        # ipdb.set_trace()
    else:
        logging.info("damn")
        pass
        # if normalise_rewards: # RL w normalised rewards
        #     assert true_reward_stats is not None, "You told me to normalise rewards for RL but you haven't specified mean and variance of reward function w.r.t. examples in prefs_buffer!"
        #     rt_mean, rt_var = true_reward_stats
        #     rew = (true_reward - rt_mean) / np.sqrt(rt_var + 1e-8)
        # else: # RL wo normalised rewards
        #     rew = true_reward

    q_values         = q_net(state)
    next_q_values    = q_target(next_state).detach() # params from target network held fixed when optimizing loss func

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # see https://colab.research.google.com/drive/1-6aNmf16JcytKw3BJ2UfGq5zkik1QLFm or https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    next_q_value, _  = next_q_values.max(-1) # max returns a (named)tuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
    expected_q_value = rew + q_net.gamma * next_q_value * (1 - done)
    
    # TODO clip the error term
    loss = (q_value - expected_q_value).pow(2).mean() # mean is across batch dimension
    return loss


def init_agent(args):
    """Intitialises and returns the necessary objects for
       Deep Q-learning:
       Q-network, target network, replay buffer and optimizer.
    """
    if args.dqn_archi == 'mlp':
        q_net = DQN(args.obs_shape, args.n_actions, args)
        q_target = DQN(args.obs_shape, args.n_actions, args)
    elif args.dqn_archi == 'conv':
        q_net = CnnDQN(args.obs_shape, args.n_actions, args)
        q_target = CnnDQN(args.obs_shape, args.n_actions, args)
    q_target.load_state_dict(q_net.state_dict()) # set params of q_target to be the same
    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    optimizer_agent = optim.Adam(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent)
    return q_net, q_target, replay_buffer, optimizer_agent

class LinearSchedule(object):
    """Copy-pasted from:
       https://github.com/openai/baselines/blob/master/baselines/common/schedules.py
    """
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)