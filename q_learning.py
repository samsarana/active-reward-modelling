"""Classes and functions to do Q-learning"""

import math, random, argparse, sys, time, itertools
import numpy as np
from collections import deque
from tqdm import tqdm, trange
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
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)


class AgentExperience():
    """For collecting experience from rollouts in a way that is
       friendly to downstream processes.
       add(sa_pair): 
       In particular, AgentExperience() instances are tensors
       with size of dim 1 that can be spe
    """
    def __init__(self, experience_shape, use_indiff_labels=True):
        self.num_clips, self.clip_length, self.obs_act_size = experience_shape
        self.use_indiff_labels = use_indiff_labels
        self.clips = np.zeros(shape=experience_shape) # default dtype=np.float64. OK for torching later?
        self.clip_rewards = np.zeros(shape=(self.num_clips, self.clip_length))
        self.clip_returns = np.zeros(shape=self.num_clips) # TODO remove as it's unused, apart from as a check
        self.i = 0 # maintain pointer to where to add next clip

    def add(self, oa_pair, reward):
        """Takes oa_pair of type torch.tensor(dtype=torch.float)
           and adds it to the current clip
           (or the next clip if the current clip is full)
           Also adds corresponding reward to return of current clip
           self.clips.shape = num_clips, clip_length, obs_act_size
           self.clip_returns.shape = num_clips
        """
        assert len(oa_pair) == self.obs_act_size
        i_clip = self.i // self.clip_length
        i_step = self.i % self.clip_length
        try:
            self.clips[i_clip, i_step] = oa_pair
            self.clip_rewards[i_clip, i_step] = reward
            self.clip_returns[i_clip] += reward
            self.i += 1 # increment pointer
        except IndexError:
            print('Oopsie, agent_experience buffer (self.clips) is full!')

    def sample(self, batch_size):
        """Samples, without replacement, batch_size *pairs* of clips
           i.e. 2 * `batch_size` clips in total
           Returns batch of pairs (shape=batch_size, 2, clip_length, obs_act_length)
           and mu in {0,1} where mu=1 if R(clip1) > R(clip2) else 0.
           If we were learning from human preferences, we wouldn't have access to R,
           but we are instead synthetically generating the preference mu from
           our access to GT reward (which is hidden from the agent).
           **Assumption: when sampling, self.clips is full**
        """
        assert self.i == self.num_clips * self.clip_length # check Assumption
        assert self.clips.shape[0] >= batch_size*2, "Trying to sample {} clip pairs but agent_experience only has {} clips!".format(batch_size*2, self.clips.shape[0])
        rows_i = np.random.choice(batch_size*2, size=(batch_size,2), replace=False)
        clip_pairs = self.clips[rows_i] # TODO fancy indexing is slow. is this a bottleneck?
        rewards = self.clip_rewards[rows_i]
        returns = self.clip_rewards[rows_i].sum(axis=-1)
        returns2 = self.clip_returns[rows_i] # TODO remove clip_returns as an attr of AgentExperience; it's just wasting computation
        assert (returns == returns2).all()
        if self.use_indiff_labels:
            mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                           np.where(returns[:, 0] == returns[:, 1], 0.5, 0))
        else:
            mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                           np.where(returns[:, 0] == returns[:, 1], random.choice([0, 1]), 0))
            # mus = returns[:, 0] > returns[:, 1] # old version that didn't handle indifference
        return clip_pairs, rewards, mus


def q_learning_loss(q_net, q_target, replay_buffer,
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
    batch_size = min(len(replay_buffer), q_net.batch_size) # TODO is this line time-consuming?
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # compute r_hats according to current reward_model and/or normalise rewards
    normalise_rewards = True if mean_rew and var_rew else False
    if normalise_rewards:
        if reward_model:
            sa_pair = torch.cat((state, action.unsqueeze(1).float()), dim=1)
            assert isinstance(reward_model, nn.Module)
            reward_model.eval() # turn off dropout at 'test' time i.e. when getting rewards to send to DQN
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


def compute_mean_var(r, prefs_buffer):
    """Given reward function r and an instance of PrefsBuffer,
       returns E[r(s,a)] and Var[r(s,a)]
       where the expectation and variance are over all the (s,a) pairs
       currently in the buffer (prefs_buffer.clip_pairs).
       The returned scalars are python numbers
    """
    assert isinstance(r, nn.Module)
    # flatten the clip_pairs and chuck them through the reward function
    sa_pairs = prefs_buffer.all_flat_sa_pairs()
    r.eval() # turn off dropout
    rews = r(sa_pairs).squeeze()
    assert rews.shape == (prefs_buffer.current_length * 2 * prefs_buffer.clip_length,)
    return rews.mean().item(), rews.var().item()


def do_RL(env, q_net, q_target, optimizer_agent, replay_buffer, num_clips, reward_model, prefs_buffer, args, i_train_round, dummy_ep_length, obs_shape, act_shape, writer1, writer2):
    # compute mean and variance of true and predicted reward (for normalising rewards sent to agent)
    rp_mean, rp_var = compute_mean_var(reward_model, prefs_buffer)
    rt_mean, rt_var = prefs_buffer.compute_mean_var_GT()
    # bookkeeping
    agent_experience = AgentExperience((num_clips, args.clip_length, obs_shape+act_shape), args.use_indiff_labels) # since episodes do not end we will collect one long trajectory then sample clips from it     
    dummy_returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
                    'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}
    
    # Train agent for args.n_agent_steps
    state = env.reset()
    with trange(args.n_agent_steps) as t:
        t.set_description('Stage 1.1: RL using reward model for {} agent steps'.format(args.n_agent_steps))
        for step in t:
            # agent interact with env
            action = q_net.act(state, q_net.epsilon)
            assert env.action_space.contains(action)
            next_state, r_true, _, _ = env.step(action) # one continuous episode
            # record step infomration
            sa_pair = torch.tensor(np.append(state, action)).float()
            agent_experience.add(sa_pair, r_true) # include reward in order to later produce synthetic prefs
            replay_buffer.push(state, action, r_true, next_state, False) # reward used to check against RL baseline; done=False since agent is in one continuous episode
            dummy_returns['ep']['true'] += r_true
            dummy_returns['ep']['true_norm'] += (r_true - rt_mean) / np.sqrt(rt_var + 1e-8) # TODO make a custom class for this, np array with size fixed in advance, given 2 means and vars, have it do the normalisation automatically and per batch (before logging)
            # also log reward the agent thinks it's getting according to current reward_model
            if not args.RL_baseline:
                reward_model.eval() # dropout off at 'test' time i.e. when logging performance
                r_pred = reward_model(sa_pair).item() # TODO ensure that this does not effect gradient computation for reward_model in stage 1.3
                dummy_returns['ep']['pred'] += r_pred
                dummy_returns['ep']['pred_norm'] += (r_pred - rp_mean) / np.sqrt(rp_var + 1e-8)
            # prepare for next step
            state = next_state

            # log performance after a "dummy" episode has elapsed
            if step % dummy_ep_length == 0 or step == args.n_agent_steps - 1:
                if not args.RL_baseline:
                    writer1.add_scalar('dummy ep return against step/round {}'.format(i_train_round), dummy_returns['ep']['pred'], step)
                    writer1.add_scalar('normalised dummy ep return against step/round {}'.format(i_train_round), dummy_returns['ep']['pred_norm'], step)
                # interpreting writers: 2 == blue == true!
                writer2.add_scalar('dummy ep return against step/round {}'.format(i_train_round), dummy_returns['ep']['true'], step)
                writer2.add_scalar('normalised dummy ep return against step/round {}'.format(i_train_round), dummy_returns['ep']['true_norm'], step)
                for key, value in dummy_returns['ep'].items():
                    dummy_returns['all'][key].append(value)
                    dummy_returns['ep'][key] = 0

            # update q_target
            if step % args.target_update_period == 0:
                q_target.load_state_dict(q_net.state_dict())

            # q_net gradient step
            if step % args.agent_gdt_step_period == 0:
                if args.RL_baseline:
                    loss_agent = q_learning_loss(q_net, q_target, replay_buffer, rt_mean, rt_var)
                else:
                    loss_agent = q_learning_loss(q_net, q_target, replay_buffer, rp_mean, rp_var, reward_model)
                optimizer_agent.zero_grad()
                loss_agent.backward()
                optimizer_agent.step()
                # decay epsilon every learning step
                if agent.epsilon > agent.epsilon_stop:
                    agent.epsilon *= agent.epsilon_decay
                # t.set_postfix(loss=loss_agent) # log with tqdm
                writer1.add_scalar('agent loss/round {}'.format(i_train_round), loss_agent, step)
                # scheduler.step() # Ibarz doesn't mention lr annealing...

    # log mean recent return this training round
    mean_dummy_true_returns = np.sum(np.array(dummy_returns['all']['true'][-100:])) / 100. # 100 dummy eps is the final 100*200/10^5 == 1/5 eps in the round
    writer2.add_scalar('mean dummy ep returns per training round', mean_dummy_true_returns, i_train_round)
    if not args.RL_baseline:
        mean_dummy_pred_returns = np.sum(np.array(dummy_returns['all']['pred'][-100:])) / 100.
        writer1.add_scalar('mean dummy ep returns per training round', mean_dummy_pred_returns, i_train_round)
    
    return q_net, replay_buffer, agent_experience