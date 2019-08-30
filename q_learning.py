"""Classes and functions to do Q-learning"""

import gym, random, time, logging
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from reward_learning import RewardModelEnsemble, init_rm
from utils import LinearSchedule, ExpSchedule

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, args):
        super().__init__()
        if args.h3_agent: # 3 hidden layer DQN
            self.layers = nn.Sequential(
                nn.Linear(num_inputs, args.h1_agent),
                nn.ReLU(),
                nn.Linear(args.h1_agent, args.h2_agent),
                nn.ReLU(),
                nn.Linear(args.h2_agent, args.h3_agent),
                nn.ReLU(),
                nn.Linear(args.h3_agent, num_actions)
            )
        else: # 2 hidden layer DQN
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


class CnnDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, args):
        super().__init__()
        self.num_actions = num_actions
        self.batch_size = args.batch_size_agent
        self.gamma = args.gamma
        self.tau = args.target_update_tau
        # self.obs_shape = args.obs_shape_all
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), # num_inputs == env.observation_space.shape[0] == (84,84,4)[0]. Still not sure this is going to work -- can other 2 input dims be left implicitly wtih Conv2d layers? Maybe need to use Conv3d..?
            nn.ReLU(), # TODO should we use dropout and/or batchnorm in between conv layers, as in reward model?
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=7, stride=1),
            nn.ReLU(), # TODO should there be a ReLU here? (just added) it. NB notebook has no ReLUs...???
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x.view(-1, 3, 84, 84)
        x = self.convolutions(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
    
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


def q_learning_loss(q_net, q_target, replay_buffer, args, reward_model_ensemble=None,
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
    state, action, true_reward, next_state, done = replay_buffer.sample(q_net.batch_size)
    # compute r_hats according to current reward_model_ensemble and/or normalise rewards
    if reward_model_ensemble: # RL from preferences
        # if args.reinit_rm_when_q_learning:
        #     logging.info("Using reinitialised reward model to compute rewards. AGENT SHOULD NOT TRAIN!")
        #     reward_model_ensemble, _ = init_rm(args)
        action_one_hot = F.one_hot(action, num_classes=4)
        assert state.shape == (q_net.batch_size, args.obs_shape)
        assert action_one_hot.shape == (q_net.batch_size, args.act_shape)
        sa_pair = torch.cat((state, action_one_hot.float()), dim=1) # dim 0 = batch, dim 1 = state-action
        assert sa_pair.shape == (q_net.batch_size, args.obs_act_shape)
        if args.no_ensemble_for_reward_pred:
            raise NotImplementedError("You haven't updated this for reward_model of type list")
        else:
            r_preds = []
            for reward_model in reward_model_ensemble:
                reward_model.eval() # dropout off at 'test' time i.e. when logging performance
                r_preds.append(reward_model(sa_pair, normalise=normalise_rewards).detach())
            rew = torch.cat(r_preds, dim=1).mean(dim=1) # each element in list r_preds has shape (B, args.size_rm_ensemble). cat and mean along ensemble dimension
            assert rew.shape == (q_net.batch_size,)
    else:
        # logging.debug("Using TRUE REWARD")
        if normalise_rewards: # RL w normalised rewards
            assert true_reward_stats is not None, "You told me to normalise rewards for RL but you haven't specified mean and variance of reward function w.r.t. examples in prefs_buffer!"
            # rt_mean, rt_var = true_reward_stats
            # rew = (true_reward - rt_mean) / np.sqrt(rt_var + 1e-8)
            rew = (true_reward - true_reward_stats.mean) / np.sqrt(true_reward_stats.var + 1e-8)
        else: # RL wo normalised rewards
            rew = true_reward

    q_values         = q_net(state)
    next_q_values    = q_target(next_state).detach() # params from target network held fixed when optimizing loss func

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze() # see https://colab.research.google.com/drive/1-6aNmf16JcytKw3BJ2UfGq5zkik1QLFm or https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    next_q_value, _  = next_q_values.max(-1) # max returns a (named)tuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
    expected_q_value = rew + q_net.gamma * next_q_value * (1 - done)
    
    # TODO perhaps clip the error term?
    if args.dqn_loss == 'mse':
        loss = F.mse_loss(q_value, expected_q_value)
        # loss = (q_value - expected_q_value).pow(2).mean() # mean is across batch dimension
    else:
        assert args.dqn_loss == 'huber'
        loss = F.smooth_l1_loss(q_value, expected_q_value)
    return loss


def init_agent(args):
    """Intitialises and returns the necessary objects for
       Deep Q-learning:
       Q-network, target network, replay buffer and optimizer.
    """
    logging.info("Initialisaling DQN with architecture {} and optimizer {}".format(args.dqn_archi, args.optimizer_agent))
    if args.dqn_archi == 'mlp':
        q_net = DQN(args.obs_shape, args.n_actions, args)
        q_target = DQN(args.obs_shape, args.n_actions, args)
    elif args.dqn_archi == 'cnn':
        q_net = CnnDQN(args.obs_shape, args.n_actions, args)
        q_target = CnnDQN(args.obs_shape, args.n_actions, args)
    if args.optimizer_agent == 'RMSProp':
        optimizer_agent = optim.RMSprop(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent)
    else:
        assert args.optimizer_agent == 'Adam'
        optimizer_agent = optim.Adam(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent)
    if args.path_to_agent_state_dict:
        logging.info("Network parameters will be loaded from from {} rather than being initialised from scratch".format(
            args.path_to_agent_state_dict))
        checkpoint = torch.load(args.path_to_agent_state_dict)
        q_net.load_state_dict(checkpoint['policy_state_dict'])
        # optimizer_agent.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        # TODO may well need to load state dict of optimizer too, in order to continue training...
    q_target.load_state_dict(q_net.state_dict()) # set params of q_target to be the same
    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    if args.epsilon_annealing_scheme == 'linear':
        epsilon_schedule = LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.n_agent_steps),
                                      initial_p=args.epsilon_start,
                                      final_p=args.epsilon_stop)
    else:
        assert args.epsilon_annealing_scheme == 'exp'
        epsilon_schedule = ExpSchedule(decay_rate=args.epsilon_decay, final_p=args.epsilon_stop, initial_p=args.epsilon_start)

    return q_net, q_target, replay_buffer, optimizer_agent, epsilon_schedule