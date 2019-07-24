"""Classes and functions to collect and annotate agent experience,
   in order to do reward learning.
"""

import numpy as np
import logging
from itertools import combinations
from active_learning import *

def acquire_labels_by_index(rand_clip_pairs, num_labels, args, reward_model):
    if args.acq_func == 'random':
        n_clip_pairs, _, _, _ = rand_clip_pairs.shape
        idx = np.random.choice(n_clip_pairs, size=num_labels, replace=False)
        info_per_clip_pair = None
    else:
        info_per_clip_pair = args.acq_func(rand_clip_pairs, reward_model, args)
        idx = np.argpartition(info_per_clip_pair, -num_labels)[-num_labels:].numpy() # see: tinyurl.com/ya7xr4kn
    return idx, info_per_clip_pair


def generate_rand_clip_pairing(agent_experience, num_labels_requested, args):
    if args.acq_search_strategy == 'christiano':
        logging.info('Collecting {} clip pairs and selecting the best 1/{} using {} acquisition func'.format(
                    args.selection_factor * num_labels_requested, args.selection_factor, args.acq_func))
        rand_clip_data = agent_experience.sample_pairs(args.selection_factor * num_labels_requested) # rand_clip_data = (rand_clip_pairs, rand_rews, rand_mus)
    elif args.acq_search_strategy == 'all_pairs':
        logging.info('Collecting all possible clip pairs. We will later select the best {} using {} acquisition func'.format(
                        num_labels_requested, args.acq_func))
        rand_clip_data = agent_experience.sample_all_pairs()
    else:
        raise NotImplementedError('You specified {} as the acq_search_strategy but I don"t know what that is!'.format(args.acq_search_strategy))
    return rand_clip_data


def log_acquisition(idx, info_per_clip_pair, clip_pairs, rews, mus, rand_clip_pairs, rand_rews, rand_mus, i_label, args, writers):
    """1. Scalar plot information gain against i_label
       2. Histogram plot informativeness of each clip pair (candidate and selected)
       3. Dumps clip pairs (candidate and selected) into csv s.t. we can view what clips are chosen
          as well as their labels and rewards
       4. Returns array [no. 0-labels, no. 1/2 labels, no. 1 labels] s.t. we can
          accumulate the frequency with which each label is acquired.
    """
    writer1, writer2 = writers
    if args.active_method:
        assert info_per_clip_pair is not None
        assert len(info_per_clip_pair.shape) == 1
        total_info = info_per_clip_pair.sum()
        selected_info = info_per_clip_pair[idx].sum()
        writer1.add_scalar('5.info_gain_per_label', selected_info, i_label)
        writer2.add_scalar('5.info_gain_per_label', total_info, i_label)

        num_pairs = len(info_per_clip_pair)
        colours = ['tab:orange' if i in idx else 'tab:blue' for i in range(num_pairs)]
        info_bars = plt.figure()
        plt.title('Information gain per clip pair')
        plt.xlabel('Clip pairs')
        plt.ylabel('Metric of info gain')
        plt.bar(np.arange(num_pairs), info_per_clip_pair, color=colours)
        writer1.add_figure('3.info_gain_per_clip_pair', info_bars, i_label)

    # TODO dump pairs (candidate and selected) into csv s.t. we can view what clips are chosen
    # as well as their labels and rewards

    mu_counts = dict(Counter(mus))
    rand_mu_counts = dict(Counter(rand_mus))
    label_counts = np.array([[mu_counts.get(0, 0), mu_counts.get(0.5, 0), mu_counts.get(1, 0)],
                          [rand_mu_counts.get(0, 0), rand_mu_counts.get(0.5, 0), rand_mu_counts.get(1, 0)]
                         ])
    return label_counts

class AgentExperience():
    """For collecting experience from rollouts in a way that is
       friendly to downstream processes.
       add(sa_pair): 
       In particular, AgentExperience() instances are tensors
       with size of dim 1 that can be spe
    """
    def __init__(self, experience_shape, force_label_choice=False):
        self.num_clips, self.clip_length, self.obs_act_size = experience_shape
        self.force_label_choice = force_label_choice
        self.clips = np.zeros(shape=experience_shape) # default dtype=np.float64. OK for torching later?
        self.clip_rewards = np.zeros(shape=(self.num_clips, self.clip_length))
        # self.clip_returns = np.zeros(shape=self.num_clips) # TODO remove as it's unused, apart from as a check
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
            # self.clip_returns[i_clip] += reward
            self.i += 1 # increment pointer
        except IndexError:
            raise RuntimeError('Oopsie, agent_experience buffer (self.clips) is full!')

    def sample_singles(self, batch_size):
        """Samples, without replacement, batch_size *single* clips
           Returns batch of clips (shape=batch_size, clip_length, obs_act_length)
           and rewards (shape=batch_size, clip_length)
           Rewards are returned in order to (i) compute mu later on
           once clips are paired together, and (ii) compute mean and variance
           of reward functions over prefs buffer to normalise rewards sent to agent
           **Assumption: when sampling, self.clips is full**
        """
        assert self.i == self.num_clips * self.clip_length, "Whoops, self.clips must be full when sampling otherwise your algo is incorrect!"
        assert self.clips.shape[0] >= batch_size, "Trying to sample {} clips but agent_experience only has {} clips!".format(batch_size, self.clips.shape[0])
        rows_i = np.random.choice(batch_size, size=(batch_size,), replace=False)
        clip_pairs = self.clips[rows_i]
        rewards = self.clip_rewards[rows_i]
        return clip_pairs, rewards

    def sample_pairs(self, batch_size):
        """Samples, without replacement, batch_size *pairs* of clips
           i.e. 2 * `batch_size` clips in total
           Returns batch of pairs (shape=batch_size, 2, clip_length, obs_act_length)
           and mu in {0,1} where mu=1 if R(clip1) > R(clip2) else 0.
           If we were learning from human preferences, we wouldn't have access to R,
           but we are instead synthetically generating the preference mu from
           our access to GT reward (which is hidden from the agent).
           **Assumption: when sampling, self.clips is full**
        """
        assert self.i == self.num_clips * self.clip_length, "Whoops, self.clips must be full when sampling otherwise your algo is incorrect!"
        assert self.clips.shape[0] >= batch_size*2,\
            "Trying to sample {} clips ({} labels/clip_pairs) but agent_experience only has {} clips!".format(
            batch_size*2, batch_size, self.clips.shape[0])
        rows_i = np.random.choice(batch_size*2, size=(batch_size,2), replace=False)
        clip_pairs = self.clips[rows_i] # TODO fancy indexing is slow. is this a bottleneck?
        rewards = self.clip_rewards[rows_i]
        returns = rewards.sum(axis=-1)
        # returns2 = self.clip_returns[rows_i] # TODO remove clip_returns as an attr of AgentExperience; it's just wasting computation
        # assert (returns == returns2).all()
        if self.force_label_choice:
            mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                            np.where(returns[:, 0] == returns[:, 1], random.choice([0, 1]), 0))
        else: # allow clips to be labeled as 0.5
            mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                           np.where(returns[:, 0] == returns[:, 1], 0.5, 0))
        return clip_pairs, rewards, mus

    def sample_all_pairs(self):
        """Returns batch of pairs (shape=batch_size, 2, clip_length, obs_act_length)
           where batch_size = self.num_clips ** 2 since we are sampling all pairs.
           NB In some sense this isn't actually samping *all* pairs of clips.
              Why? When we collect clips in AgentExperience objects, the sequence of
              state-actions pairs is divided immediately into clips
              In other words, *all* pairs does not include, for example, a pair of clips
              which are offset by 1 timestep at collection time/when the agent
              is acting in the environment. Nor 2 timesteps. Only multiples of k
              timesteps, where `k = self.clip_length`.
              However, I'm not going to rewrite the whole method of storing and sampling
              from agent experience to change this; I'm not even clear which of the
              two methods I'd be more interested in anyway.
              Plus, I'm only going to use this method in order to check how
              sampling all pairs affects the performance of active learning
              relative to random acquisition. In more complex envs, it will be intractable
              to sample all pairs anyway.
              ** TODO It does seem worth checking whether the Christiano/Ibarz implemenations
                 include clips that are offset by < self.clip_length timesteps, or whether
                 they also carve up agent experience into clips as soon as it is acquired,
                 and sample from those discretised clips only...
        """
        assert self.i == self.num_clips * self.clip_length, "Whoops, self.clips should be full when sampling!"
        assert self.clips.shape[0] == self.num_clips
        all_clips_paired = np.array([[self.clips[i], self.clips[j]] for i, j in combinations(range(self.num_clips), 2)])
        rewards = np.array([[self.clip_rewards[i], self.clip_rewards[j]] for i, j in combinations(range(self.num_clips), 2)])
        returns = rewards.sum(axis=-1)
        if self.force_label_choice:
            mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                                np.where(returns[:, 0] == returns[:, 1], random.choice([0, 1]), 0))
        else: # allow clips to be labeled as 0.5
            mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                                np.where(returns[:, 0] == returns[:, 1], 0.5, 0))
        return all_clips_paired, rewards, mus
        

# OLD FUNCTION, though I'm still using it b/c I haven't refactored training_protocol() to use my new functions yet

def sample_and_annotate_clip_pairs(agent_experience, reward_model, num_labels_requested, args, writers, i_train_round):
    writer1, _ = writers
    writer1.add_scalar('6.labels_requested_per_round', num_labels_requested, i_train_round)
    if args.active_method:
        logging.info('Acquiring clips using {} acquisition function and uncertainty estimates from {}'.format(args.active_method, args.uncert_method))
        if args.acq_search_strategy == 'christiano':
            logging.info('Doing Active Learning, so actually collect {} clip pairs and select the best 1/{} using {} method'.format(
                        args.selection_factor * num_labels_requested, args.selection_factor, args.active_method))
            rand_clip_data = agent_experience.sample_pairs(args.selection_factor * num_labels_requested) # rand_clip_data = (rand_clip_pairs, rand_rews, rand_mus)
        elif args.acq_search_strategy == 'all_pairs':
            logging.info('Doing Active Learning, and collecting all possible clip pairs. Selecting the best {} using {} method'.format(
                            num_labels_requested, args.active_method))
            rand_clip_data = agent_experience.sample_all_pairs()
        else:
            raise NotImplementedError('You specified {} as the acq_search_strategy but I don"t know what that is!'.format(args.acq_search_strategy))
        clip_pairs, rews, mus, label_counts = acquire_clip_pairs(rand_clip_data, reward_model, num_labels_requested, args, writers, i_train_round)
    else:
        logging.info('Acquiring clips by random acquisition')
        clip_pairs, rews, mus = agent_experience.sample_pairs(num_labels_requested)
        label_counts = log_random_acquisitions(mus, rews, writers, args, i_train_round)
    return clip_pairs, rews, mus, label_counts