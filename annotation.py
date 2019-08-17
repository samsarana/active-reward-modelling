"""Classes and functions to collect and annotate agent experience,
   in order to do reward learning.
"""

import numpy as np
import logging
from itertools import combinations
from active_learning import *
from active_learning_logging import *

def generate_rand_clip_pairing(agent_experience, num_labels_requested, args):
    if args.acq_search_strategy == 'christiano':
        logging.info('Collecting {} clip pairs and selecting the best 1/{} using {} acquisition func'.format(
                    args.selection_factor * num_labels_requested, args.selection_factor, args.acquistion_func.__name__))

        rand_clip_data = agent_experience.sample_pairs(args.selection_factor * num_labels_requested) # rand_clip_data = (rand_clip_pairs, rand_rews, rand_mus)
    elif args.acq_search_strategy == 'all_pairs':
        logging.info('Collecting all possible clip pairs. We will later select the best {} using {} acquisition func'.format(
                        num_labels_requested, args.acquistion_func.__name__))
        rand_clip_data = agent_experience.sample_all_pairs()
    else:
        raise NotImplementedError('You specified {} as the acq_search_strategy but I don"t know what that is!'.format(args.acq_search_strategy))
    return rand_clip_data


def make_acquisitions(rand_clip_data, reward_model, prefs_buffer, args, writers, mu_counts_total, i_label):
    # Stage 1.2: Sample `batch_size_acq` clip pairs without replacement from recent rollouts and label them (synthetically)
    logging.info("Acquisition {}: acquiring {} clip pair(s) in a single batch".format(i_label, args.batch_size_acq))
    rand_clip_pairs, rand_rews, rand_mus = rand_clip_data
    idx, info_per_clip_pair = args.acquistion_func(rand_clip_pairs, args.batch_size_acq, args, reward_model)
    # idx, info_per_clip_pair = acquire_labels_by_index(rand_clip_pairs, args.batch_size_acq, args, reward_model)
    # put labelled clip_pairs into prefs_buffer and accumulate count of each label acquired
    clip_pairs, rews, mus = rand_clip_pairs[idx], rand_rews[idx], rand_mus[idx]
    prefs_buffer.push(clip_pairs, rews, mus)
    # log this acquistion
    mu_counts_acq = log_acquisition(idx, info_per_clip_pair, clip_pairs, rews, mus, rand_clip_pairs, rand_rews, rand_mus, i_label, args, writers)
    mu_counts_total += mu_counts_acq
    # remove sampled clip pairs
    rand_clip_pairs = np.delete(rand_clip_pairs, idx, axis=0)
    rand_rews = np.delete(rand_rews, idx, axis=0)
    rand_mus = np.delete(rand_mus, idx, axis=0)
    rand_clip_data = rand_clip_pairs, rand_rews, rand_mus
    return prefs_buffer, rand_clip_data, mu_counts_total


class AgentExperience():
    """For collecting experience from rollouts in a way that is
       friendly to downstream processes.
    """
    def __init__(self, num_clips, args):
        self.num_clips = num_clips
        self.clip_length = args.clip_length
        self.oa_shape = args.obs_act_shape
        self.oa_data_type = args.oa_dtype
        experience_shape = (num_clips, self.clip_length, self.oa_shape)
        self.clips = np.zeros(shape=experience_shape, dtype=self.oa_data_type)
        self.clip_rewards = np.zeros(shape=(num_clips, self.clip_length))#, dtype=self.oa_data_type)
        self.i = 0 # maintain pointer to where to add next clip
        self.force_label_choice = args.force_label_choice
        self.n_sample_reps = args.n_sample_reps

    def add(self, oa_pair, reward):
        """Takes oa_pair of type torch.tensor(dtype=torch.float)
           and adds it to the current clip
           (or the next clip if the current clip is full)
           Also adds corresponding reward to return of current clip
           self.clips.shape = num_clips, clip_length, oa_shape
           self.clip_rewards.shape = num_clips, self.clip_length
        """
        assert len(oa_pair) == self.oa_shape
        i_clip = self.i // self.clip_length
        i_step = self.i % self.clip_length
        # the following assert should be redundant due to checking same thing in main training loop
        assert oa_pair.dtype == self.oa_data_type, "Trying to add oa_pair of dtype {} to array made for dtype {}".format(oa_pair.dtype, self.oa_data_type)
        try:
            self.clips[i_clip, i_step] = oa_pair
            self.clip_rewards[i_clip, i_step] = reward
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
        if self.n_sample_reps > 1:
            logging.info('DEBUGGING: {} exact copies of the first clip pair to be sampled from AgentExperience will also be given to the acq func. Total clip pairs sampled = {}'.format(
                self.n_sample_reps, batch_size + self.n_sample_reps))
            # rows_i = np.tile(rows_i, (self.n_sample_reps, 1)) # tile the rows to sample `self.n_sample_reps` times in 0th dim (and 1 time--i.e. don't tile--in 1st dim)
            rows_i = np.concatenate((rows_i, np.tile(rows_i[0], (self.n_sample_reps,1))), axis=0) # repeat the first pair clips to be sampled `self.n_sample_reps` times (along 0th dim)
        clip_pairs = self.clips[rows_i] # TODO fancy indexing is slow. is this a bottleneck?
        rewards = self.clip_rewards[rows_i]
        returns = rewards.sum(axis=-1)
        if self.force_label_choice:
            mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                            np.where(returns[:, 0] == returns[:, 1], random.choice([0, 1]), 0))
        else: # allow clips to be labeled as 0.5 (default)
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