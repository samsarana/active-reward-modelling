"""For logging things to do with Active Learning"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def log_info_gain(info_per_clip_pair, idx, writers, round_num):
    """1. Plots a bar chart with the info of each clip pair
          Clips pairs which were selected (given by idx) are orange; rest are blue.
       2. Plots two scalars: info summed over all clip pairs
                             info summed over selected clip pairs
       All plotting is done by logging to Tensorboard.
    """
    writer1, writer2 = writers
    assert len(info_per_clip_pair.shape) == 1
    num_pairs = len(info_per_clip_pair)
    colours = ['tab:orange' if i in idx else 'tab:blue' for i in range(num_pairs)]
    info_bars = plt.figure()
    plt.title('Information gain per clip pair')
    plt.xlabel('Clip pairs')
    plt.ylabel('Metric of info gain')
    plt.bar(np.arange(num_pairs), info_per_clip_pair, color=colours)
    writer1.add_figure('3.info_gain_per_clip_pair', info_bars, round_num)

    total_info = info_per_clip_pair.sum()
    selected_info = info_per_clip_pair[idx].sum()
    writer1.add_scalar('5.info_gain_per_round', selected_info, round_num)
    writer2.add_scalar('5.info_gain_per_round', total_info, round_num)

def log_acquisitions(mus, rand_mus, rews, rand_rews, writers, args, round_num):
    """Plots two histograms: the labels and return
       for each clip pair candidate and acquisition.
    """
    writer1, writer2 = writers

    mu_counts = dict(Counter(mus))
    rand_mu_counts = dict(Counter(rand_mus))
    label_counts = np.array([[mu_counts.get(0, 0), mu_counts.get(0.5, 0), mu_counts.get(1, 0)],
                          [rand_mu_counts.get(0, 0), rand_mu_counts.get(0.5, 0), rand_mu_counts.get(1, 0)]
                         ]) # use 0 as default value from dict.get()
    # mus_, mu_counts = np.unique(mus, return_counts=True) # essentially gives us a discrete histogram
    # rand_mus_, rand_mu_counts = np.unique(rand_mus, return_counts=True)
    # assert (mus_ == [0, 0.5, 1]).all() and len(mu_counts) == 3
    # assert (rand_mus_ == [0, 0.5, 1]).all() and len(rand_mu_counts) == 3
    # label_counts = np.stack((mu_counts, rand_mu_counts)) # acquired, candidate

    labels_hist = plt.figure()
    plt.title('Label histogram, round {}'.format(round_num))
    plt.xlabel('mu')
    plt.ylabel('Frequency')
    # plt.yscale('log')
    plt.hist(rand_mus, bins=11, range=(-0.05,1.05), color='tab:blue', alpha=0.7, label='Candidate')
    plt.hist(mus, bins=11, range=(-0.05,1.05), color='tab:orange', alpha=0.7, label='Acquired')
    plt.legend()
    writer1.add_figure('1.label_histogram', labels_hist, round_num)

    mean_ret_hist = plt.figure()
    plt.title('Return histogram, round {}'.format(round_num))
    plt.xlabel('Return, averaged over both clips in pair')
    plt.ylabel('Frequency')
    # plt.yscale('log')
    if len(rand_rews.shape) == 3: # v0 acq func => rand_rews paired
        mean_rand_rews = rand_rews.sum(-1).sum(-1) / 2
    elif len(rand_rews.shape) == 2: # v1 acq func => rand_rews not paired
        mean_rand_rews = rand_rews.sum(-1)
    else:
        raise RuntimeError('`rand_rews` is of the wrong shape.')
    rews_max = args.clip_length * 1
    # as a approximation to the min reward, in CartPoleContinuous the agent never seems to do worse than ending the episode once per 4 steps
    # NB this is a very crude approximation and will definitely not transfer to other envs
    assert args.env_class == 'gym_barm:CartPoleContinuous-v0', "You ought to adjust the range of your histogram plots because your current values are tuned to CartPoleContinuous"
    rews_min = args.ep_end_penalty * args.clip_length * 1/4  + 1 * args.clip_length * (1 - 1/4)
    rand_label = 'Candidate (paired)' if args.acq_search_strategy == 'v0' else 'Candidate (unpaired)'
    plt.hist(mean_rand_rews, bins=100, range=(rews_min, rews_max), # min possible value is -39*25 
        color='tab:blue', alpha=0.7, label=rand_label)
    plt.hist(rews.sum(-1).sum(-1) / 2, bins=100, range=(rews_min, rews_max), 
        color='tab:orange', alpha=0.7, label='Acquired')
    plt.legend()
    writer1.add_figure('2.return_histogram', mean_ret_hist, round_num)

    # Tensorboard histograms are bad for discrete data but can be dynamically adjusted so I'll print them anyway as a complementary thing
    writer1.add_histogram('1.labels_acquired_and_candidate', mus, round_num, bins='auto')
    writer2.add_histogram('1.labels_acquired_and_candidate', rand_mus, round_num, bins='auto')
    writer1.add_histogram('2.mean_return_of_clip_pairs_acquired_and_candidate', rews.sum(-1).sum(-1) / 2, round_num, bins='auto')
    writer2.add_histogram('2.mean_return_of_clip_pairs_acquired_and_candidate', mean_rand_rews, round_num, bins='auto')
    return label_counts


def log_random_acquisitions(mus, rews, writers, args, round_num):
    """Plots two histograms: the labels and return
       for each clip pair acquired by random baseline.
    """
    writer1, writer2 = writers
    counts = dict(Counter(mus))
    mu_counts = np.array([counts.get(0, 0), counts.get(0.5, 0), counts.get(1, 0)]) # use 0 as default value from dict.get()
    assert mu_counts.sum() > 0
    # mus_unique, mu_counts = np.unique(mus, return_counts=True) # essentially gives us a discrete histogram
    # assert (mus_ == [0, 0.5, 1]).all() and len(mu_counts) == 3
    # writer1.add_scalar('10.mu_counts', mu_counts[0], round_num)
    # writer2.add_scalar('10.mu_counts', mu_counts[1], round_num)
    # writer3.add_scalar('10.mu_counts', mu_counts[2], round_num)

    labels_hist = plt.figure()
    plt.title('Label histogram, round {}'.format(round_num))
    plt.xlabel('mu')
    plt.ylabel('Frequency')
    # plt.yscale('log')
    plt.hist(mus, bins=11, range=(-0.05,1.05), color='tab:orange', alpha=0.7, label='Acquired')
    plt.legend()
    writer1.add_figure('1.label_histogram', labels_hist, round_num)

    mean_ret_hist = plt.figure()
    plt.title('Return histogram, round {}'.format(round_num))
    plt.xlabel('Return, averaged over both clips in pair')
    plt.ylabel('Frequency')
    # plt.yscale('log')
    rews_max = args.clip_length * 1
    # as a approximation to the min reward, in CartPoleContinuous the agent never seems to do worse than ending the episode once per 4 steps
    # NB this is a very crude approximation and will definitely not transfer to other envs
    assert args.env_class == 'gym_barm:CartPoleContinuous-v0', "You ought to adjust the range of your histogram plots because your current values are tuned to CartPoleContinuous"
    rews_min = args.ep_end_penalty * args.clip_length * 1/4  + 1 * args.clip_length * (1 - 1/4)
    rand_label = 'Candidate (paired)' if args.acq_search_strategy == 'v0' else 'Candidate (unpaired)'
    plt.hist(rews.sum(-1).sum(-1) / 2, bins=100, range=(rews_min, rews_max), 
        color='tab:orange', alpha=0.7, label='Acquired')
    plt.legend()
    writer1.add_figure('2.return_histogram', mean_ret_hist, round_num)

    # Tensorboard histograms are bad for discrete data but can be dynamically adjusted so I'll print them anyway as a complementary thing
    writer1.add_histogram('1.labels_acquired_and_candidate', mus, round_num, bins='auto')
    writer1.add_histogram('2.mean_return_of_clip_pairs_acquired_and_candidate', rews.sum(-1).sum(-1) / 2, round_num, bins='auto')
    return mu_counts


def log_total_mu_counts(mu_counts, writers, args):
    writer1, writer2 = writers
    if args.active_method:
        assert mu_counts.shape == (2,3)
        acquired, candidate = mu_counts
        writer2.add_scalar('10.total_mu_counts', candidate[0], 0)
        writer2.add_scalar('10.total_mu_counts', candidate[1], 0.5) # TODO can global step be float?
        writer2.add_scalar('10.total_mu_counts', candidate[2], 1)
    else:
        assert mu_counts.shape == (3,)
        acquired = mu_counts
    writer1.add_scalar('10.total_mu_counts', acquired[0], 0)
    writer1.add_scalar('10.total_mu_counts', acquired[1], 0.5) # TODO can global step be float?
    writer1.add_scalar('10.total_mu_counts', acquired[2], 1)