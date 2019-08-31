"""For logging things to do with Active Learning"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
import gym
from gym import wrappers
from time import time, sleep

def log_acquisition(idx, info_per_clip_pair, clip_pairs, rews, mus, rand_clip_pairs, rand_rews, rand_mus, i_label, args, writers):
    """1. Scalar plot information gain against i_label
       2. Histogram plot informativeness of each clip pair (candidate and selected)
       3. Dumps clip pairs (candidate and selected) into csv s.t. we can view what clips are chosen
          as well as their labels and rewards
       4. Returns array [no. 0-labels, no. 1/2 labels, no. 1 labels] s.t. we can
          accumulate the frequency with which each label is acquired.
       Takes:
         `clip_pairs`: batch of clip pairs (shape=batch_size, 2, clip_length, obs_act_length)
    """
    writer1, writer2 = writers
    # log no. labels of each type acquired
    mu_counts = dict(Counter(mus))
    rand_mu_counts = dict(Counter(rand_mus))
    label_counts = np.array([[mu_counts.get(0, 0), mu_counts.get(0.5, 0), mu_counts.get(1, 0)],
                          [rand_mu_counts.get(0, 0), rand_mu_counts.get(0.5, 0), rand_mu_counts.get(1, 0)]
                         ])
    rew_counts = dict(Counter(rews.reshape(-1)))
    rand_rew_counts = dict(Counter(rand_rews.reshape(-1)))
    writer1.add_scalar('5c.0_labels', mu_counts.get(0, 0), i_label)
    writer1.add_scalar('5c.0.5_labels', mu_counts.get(0.5, 0), i_label)
    writer1.add_scalar('5c.1_labels', mu_counts.get(1, 0), i_label)
    writer2.add_scalar('5c.0_labels', rand_mu_counts.get(0, 0), i_label)
    writer2.add_scalar('5c.0.5_labels', rand_mu_counts.get(0.5, 0), i_label)
    writer2.add_scalar('5c.1_labels', rand_mu_counts.get(1, 0), i_label)

    with open('./logs/acqs.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['i_label', i_label])
        csv_writer.writerow(['idx'])
        csv_writer.writerow(idx)
        csv_writer.writerow(['STATISTICS'])
        actions = clip_pairs[...,-args.act_shape:]
        rand_actions = rand_clip_pairs[...,-args.act_shape:]
        # sum_actions      = actions.sum(axis=(0,1,2))
        # sum_rand_actions = rand_actions.sum(axis=(0,1,2))
        # csv_writer.writerow(['rews.var'])
        # csv_writer.writerow([rews.var()])
        # csv_writer.writerow(['rand_rews.var'])
        # csv_writer.writerow([rand_rews.var()])
        # csv_writer.writerow(['sum_actions'])
        # csv_writer.writerow(sum_actions)
        # csv_writer.writerow(['sum_rand_actions'])
        # csv_writer.writerow(sum_rand_actions)
        csv_writer.writerow(['ACQUIRED'])
        csv_writer.writerow(['rews_0_count'])
        csv_writer.writerow([rew_counts.get(0, 0)])
        csv_writer.writerow(['rews_1_count'])
        csv_writer.writerow([rew_counts.get(1, 0)])
        csv_writer.writerow(['rews_-1_count'])
        csv_writer.writerow([rew_counts.get(-1, 0)])
        csv_writer.writerow(['actions'])
        csv_writer.writerow(actions)
        csv_writer.writerow(['rews'])
        csv_writer.writerow(rews)
        csv_writer.writerow(['mus'])
        csv_writer.writerow(mus)
        csv_writer.writerow(['CANDIDATE'])
        csv_writer.writerow(['rand_rews_0_count'])
        csv_writer.writerow([rand_rew_counts.get(0, 0)])
        csv_writer.writerow(['rews_1_count'])
        csv_writer.writerow([rand_rew_counts.get(1, 0)])
        csv_writer.writerow(['rews_-1_count'])
        csv_writer.writerow([rand_rew_counts.get(-1, 0)])
        csv_writer.writerow(['rand_actions'])
        csv_writer.writerow(rand_actions)
        csv_writer.writerow(['rand_rews'])
        csv_writer.writerow(rand_rews)
        csv_writer.writerow(['rand_mus'])
        csv_writer.writerow(rand_mus)


    if args.active_method:
        assert info_per_clip_pair is not None
        assert len(info_per_clip_pair.shape) == 1
        total_info = info_per_clip_pair.sum()
        selected_info = info_per_clip_pair[idx].sum()
        writer1.add_scalar('5b.info_gain_per_label', selected_info, i_label)
        writer2.add_scalar('5b.info_gain_per_label', total_info, i_label)
        num_pairs = len(info_per_clip_pair)
        colours = ['tab:orange' if i in idx else 'tab:blue' for i in range(num_pairs)]
        info_bars = plt.figure()
        plt.title('Information gain per clip pair')
        plt.xlabel('Clip pairs')
        plt.ylabel('Metric of info gain')
        plt.bar(np.arange(num_pairs), info_per_clip_pair, color=colours)
        writer1.add_figure('3.info_gain_per_clip_pair', info_bars, i_label)
    return label_counts

def log_total_mu_counts(mu_counts, writers, args):
    writer1, writer2 = writers
    assert mu_counts.shape == (2,3)
    acquired, candidate = mu_counts
    writer1.add_scalar('5a.total_mu_counts', acquired[0], 0)
    writer1.add_scalar('5a.total_mu_counts', acquired[1], 1)
    writer1.add_scalar('5a.total_mu_counts', acquired[2], 2)
    
    writer2.add_scalar('5a.total_mu_counts', candidate[0], 0)
    writer2.add_scalar('5a.total_mu_counts', candidate[1], 1) # global step cannot be float so make 0.5 -> 1 and 1 -> 2
    writer2.add_scalar('5a.total_mu_counts', candidate[2], 2)