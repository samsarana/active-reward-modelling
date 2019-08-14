"""For logging things to do with Active Learning"""

import numpy as np
import matplotlib.pyplot as plt
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
    if args.active_method:
        assert info_per_clip_pair is not None
        assert len(info_per_clip_pair.shape) == 1
        total_info = info_per_clip_pair.sum()
        selected_info = info_per_clip_pair[idx].sum()
        writer1.add_scalar('5b.info_gain_per_label', selected_info, i_label)
        writer2.add_scalar('5b.info_gain_per_label', total_info, i_label)
        # num_pairs = len(info_per_clip_pair)
        # colours = ['tab:orange' if i in idx else 'tab:blue' for i in range(num_pairs)]
        # info_bars = plt.figure()
        # plt.title('Information gain per clip pair')
        # plt.xlabel('Clip pairs')
        # plt.ylabel('Metric of info gain')
        # plt.bar(np.arange(num_pairs), info_per_clip_pair, color=colours)
        # writer1.add_figure('3.info_gain_per_clip_pair', info_bars, i_label)

    mu_counts = dict(Counter(mus))
    rand_mu_counts = dict(Counter(rand_mus))
    label_counts = np.array([[mu_counts.get(0, 0), mu_counts.get(0.5, 0), mu_counts.get(1, 0)],
                          [rand_mu_counts.get(0, 0), rand_mu_counts.get(0.5, 0), rand_mu_counts.get(1, 0)]
                         ])
    # log no. labels of each type acquired
    writer1.add_scalar('5c.0_labels', mu_counts.get(0, 0), i_label)
    writer1.add_scalar('5c.0.5_labels', mu_counts.get(0.5, 0), i_label)
    writer1.add_scalar('5c.1_labels', mu_counts.get(1, 0), i_label)
    writer2.add_scalar('5c.0_labels', rand_mu_counts.get(0, 0), i_label)
    writer2.add_scalar('5c.0.5_labels', rand_mu_counts.get(0.5, 0), i_label)
    writer2.add_scalar('5c.1_labels', rand_mu_counts.get(1, 0), i_label)

    # save video of clip pairs. TODO this code doesn't work. the video is only of the first frame
    if args.save_pair_videos:
        batch_size, _, clip_length, obs_act_length = clip_pairs.shape
        for i_batch in range(batch_size):
            for pair_num in range(2):
                clip = clip_pairs[i_batch][pair_num]
                env = gym.make(args.env_ID)
                fname = '{}/videos/clip_pairs/i_label={}i_batch={}pair={}time={}/'.format(
                    args.logdir, i_label, i_batch, pair_num, str(time()))
                # debugging START
                # import ipdb
                # ipdb.set_trace()
                # debugging END
                env = wrappers.Monitor(env, fname)
                # env._before_reset()
                env.reset()
                # env._flush()
                env.reset_video_recorder()
                # env._after_reset()
                for step in range(clip_length):
                    obs = clip[step][:args.obs_shape]
                    action = clip[step][args.obs_shape:]
                    if 'Acrobot' in args.env_ID:
                        theta1 = np.arccos(obs[0])
                        if obs[1] < 0: # sin(theta1) > 0
                            theta1 = -theta1
                        theta2 = np.arccos(obs[2])
                        if obs[3] < 0: # sin(theta2) > 0
                            theta2 = -theta2
                        state = np.array([theta1, theta2, obs[4], obs[5]])
                    elif 'CartPole' in args.env_ID:
                        state = obs
                    else:
                        raise NotImplementedError("You haven't told me how to map observations to states for environment {}".format(args.env_ID))
                    # env._before_step(action)
                    # import ipdb
                    # ipdb.set_trace()
                    env.state = state
                    # env._flush(force=True)
                    env.video_recorder.capture_frame()
                    # env._after_step(obs, rews[i_batch][pair_num][step], False, {})
                    # sleep(1e-3)
                env.close()
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