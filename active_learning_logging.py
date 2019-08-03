"""For logging things to do with Active Learning"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
        # num_pairs = len(info_per_clip_pair)
        # colours = ['tab:orange' if i in idx else 'tab:blue' for i in range(num_pairs)]
        # info_bars = plt.figure()
        # plt.title('Information gain per clip pair')
        # plt.xlabel('Clip pairs')
        # plt.ylabel('Metric of info gain')
        # plt.bar(np.arange(num_pairs), info_per_clip_pair, color=colours)
        # writer1.add_figure('3.info_gain_per_clip_pair', info_bars, i_label)

    # TODO dump pairs (candidate and selected) into csv s.t. we can view what clips are chosen
    # as well as their labels and rewards

    mu_counts = dict(Counter(mus))
    rand_mu_counts = dict(Counter(rand_mus))
    label_counts = np.array([[mu_counts.get(0, 0), mu_counts.get(0.5, 0), mu_counts.get(1, 0)],
                          [rand_mu_counts.get(0, 0), rand_mu_counts.get(0.5, 0), rand_mu_counts.get(1, 0)]
                         ])
    # log no. labels of each type acquired
    writer1.add_scalar('5b.0_labels', mu_counts.get(0, 0), i_label)
    writer1.add_scalar('5b.0.5_labels', mu_counts.get(0.5, 0), i_label)
    writer1.add_scalar('5b.1_labels', mu_counts.get(1, 0), i_label)
    return label_counts

def log_total_mu_counts(mu_counts, writers, args):
    writer1, writer2 = writers
    assert mu_counts.shape == (2,3)
    acquired, candidate = mu_counts
    writer1.add_scalar('4.total_mu_counts', acquired[0], 0)
    writer1.add_scalar('4.total_mu_counts', acquired[1], 1)
    writer1.add_scalar('4.total_mu_counts', acquired[2], 2)
    
    writer2.add_scalar('4.total_mu_counts', candidate[0], 0)
    writer2.add_scalar('4.total_mu_counts', candidate[1], 1) # global step cannot be float so make 0.5 -> 1 and 1 -> 2
    writer2.add_scalar('4.total_mu_counts', candidate[2], 2)