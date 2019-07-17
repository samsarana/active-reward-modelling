"""For logging things to do with Active Learning"""

import numpy as np
import matplotlib.pyplot as plt

def log_info_gain(info_per_clip_pair, idx, writers, round_num):
    """1. Plots a bar chart with the info of each clip pair
          Clips pairs which were selected (given by idx) are orange; rest are blue.
       2. Plots two scalars: info summed over all clip pairs
                             info summed over selected clip pairs
       All plotting is done by logging to Tensorboard.
    """
    writer1, writer2, _ = writers
    assert len(info_per_clip_pair.shape) == 1
    num_pairs = len(info_per_clip_pair)
    colours = ['tab:orange' if i in idx else 'tab:blue' for i in range(num_pairs)]
    info_bars = plt.figure()
    plt.title('Information gain per clip pair')
    plt.xlabel('Clip pairs')
    plt.ylabel('Metric of info gain')
    plt.bar(np.arange(num_pairs), info_per_clip_pair, color=colours)
    writer1.add_figure('3.info gain per clip pair', info_bars, round_num)

    total_info = info_per_clip_pair.sum()
    selected_info = info_per_clip_pair[idx].sum()
    writer1.add_scalar('5.info_gain_per_round', selected_info, round_num)
    writer2.add_scalar('5.info_gain_per_round', total_info, round_num)

def log_acquisitions(mus, rand_mus, rews, rand_rews, writers, args, round_num):
    """Plots two histograms: the labels and return
       for each clip pair candidate and acquisition.
    """
    writer1, writer2, _ = writers
    labels_hist = plt.figure()
    plt.title('Label histogram, round {}'.format(round_num))
    plt.xlabel('mu')
    plt.ylabel('Frequency')
    # plt.yscale('log')
    plt.hist(rand_mus, bins=11, range=(-0.05,1.05), color='tab:blue', alpha=0.7, label='Candidate')
    plt.hist(mus, bins=11, range=(-0.05,1.05), color='tab:orange', alpha=0.7, label='Acquired')
    plt.legend()
    writer1.add_figure('1.label histogram', labels_hist, round_num)

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
    writer1.add_figure('2.return histogram', mean_ret_hist, round_num)

    # Tensorboard histograms are bad for discrete data but can be dynamically adjusted so I'll print them anyway as a complementary thing
    writer1.add_histogram('1.labels acquired and candidate', mus, round_num, bins='auto')
    writer2.add_histogram('1.labels acquired and candidate', rand_mus, round_num, bins='auto')
    writer1.add_histogram('2.mean return of clip pairs acquired and candidate', rews.sum(-1).sum(-1) / 2, round_num, bins='auto')
    writer2.add_histogram('2.mean return of clip pairs acquired and candidate', mean_rand_rews, round_num, bins='auto')


def log_random_acquisitions(mus, rews, writers, args, round_num):
    """Plots two histograms: the labels and return
       for each clip pair acquired by random baseline.
    """
    writer1, writer2, writer3 = writers
    labels_hist = plt.figure()
    plt.title('Label histogram, round {}'.format(round_num))
    plt.xlabel('mu')
    plt.ylabel('Frequency')
    # plt.yscale('log')
    plt.hist(mus, bins=11, range=(-0.05,1.05), color='tab:orange', alpha=0.7, label='Acquired')
    plt.legend()
    writer1.add_figure('1.label histogram', labels_hist, round_num)

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
    writer1.add_figure('2.return histogram', mean_ret_hist, round_num)

    # Tensorboard histograms are bad for discrete data but can be dynamically adjusted so I'll print them anyway as a complementary thing
    writer1.add_histogram('1.labels acquired and candidate', mus, round_num, bins='auto')
    writer1.add_histogram('2.mean return of clip pairs acquired and candidate', rews.sum(-1).sum(-1) / 2, round_num, bins='auto')