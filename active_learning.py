import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from reward_model import RewardModelEnsemble

def acquire_clip_pairs_v0(agent_experience, reward_model, num_labels_requested, args, writer1, writer2, i_train_round):    
    """NB I haven't tested this function since changing a bunch of things
       in the acquisitions functions (when I wrote `acquire_clip_pairs_v1`)
       1. Samples m = `args.selection_factor * num_labels_requested` *pairs*
          of clips from agent_experience.
       2. Returns batch of (sigma_1, sigma_2) that maximise 
          some metric of information gain.
          Batch size is `num_labels_requested`. Also returns corresponding
          rewards/mu for each clip/pair.
          clip_pairs.shape == (num_labels_requested, 2, args.clip_length, args.obs_act_shape)
          rews.shape       == (num_labels_requested, 2, args.clip_length)
          mus.shape        == (num_labels_requested,)
    """
    print('Doing Active Learning, so actually collect {} clip pairs and select the best 1/{} using {} method'.format(
        args.selection_factor * num_labels_requested, args.selection_factor, args.active_method))
    rand_clip_pairs, rand_rews, rand_mus = agent_experience.sample_pairs(args.selection_factor * num_labels_requested)
    if args.active_method == 'BALD':
        info_per_clip_pair = compute_info_gain(rand_clip_pairs, reward_model, args)
    elif args.active_method == 'var_ratios':
        info_per_clip_pair = compute_var_ratio(rand_clip_pairs, reward_model, args)
    elif args.active_method == 'max_entropy':
        info_per_clip_pair = compute_pred_entropy(rand_clip_pairs, reward_model, args)
    elif args.active_method == 'naive_variance':
        info_per_clip_pair = compute_sample_var_clip_pair(rand_clip_pairs, reward_model, args)
    else:
        raise RuntimeError("You specified {} as the active_method type, but I don't know what that is!".format(args.active_method))
    idx = np.argpartition(info_per_clip_pair, -num_labels_requested)[-num_labels_requested:] # see: tinyurl.com/ya7xr4kn
    clip_pairs, rews, mus = rand_clip_pairs[idx], rand_rews[idx], rand_mus[idx] # returned indices are not sorted
    log_info_gain(info_per_clip_pair, idx, writer1, writer2, round_num=i_train_round)
    log_acquisitions(mus, rand_mus, rews, rand_rews, writer1, writer2, round_num=i_train_round)
    return clip_pairs, rews, mus


def acquire_clip_pairs_v1(agent_experience, reward_model, num_labels_requested, args, writer1, writer2, i_train_round):
    """1. Samples m = `2 * args.selection_factor * num_labels_requested` clips from agent_experience.
          We double the selection factor in order to make a fair comparison with `acquire_clip_pairs_v0`
          That function samples `args.selection_factor * num_labels_requested` clip *pairs* so if we didn't
          double here, then v0 would unfairly be sampling more clips overall
       2. Finds the clip with minimum uncertainty according to current reward_model (using the sample
          variance of multiple MC samples. See Yarin's thesis p.49, 51 for why this is justified.
          TODO 1. check that I am allowed to ignore the precision term when eval'ing uncertainty of 
          different x under the same model). This will be the reference clip `sigma_1`.
       3. For every other clip `sigma_2` in agent_experience, computes the mutual information
          between the predicted label of (`sigma_1`, `sigma_2`) and `reward_model` parameters.
          (This is for BALD. If `args.active_method != 'bald'` then use some other method.)
       4. Return batch of (sigma_1, sigma_2, mu) with the sigma_2's that maximise the MI.
          Batch size is `num_labels_requested`. Also returns corresponding rewards and
          mu for each clip/pair.
          clip_pairs.shape == (num_labels_requested, 2, args.clip_length, args.obs_act_shape)
          rews.shape       == (num_labels_requested, 2, args.clip_length)
          mus.shape        == (num_labels_requested,)
          TODO 2. Work out if it's a bad idea to have a single reference clip in each example.
          Perhaps the dataset won't be diverse enough and if we request `num_labels_requested` labels
          then we should have that number of reference clips, and find a high-information pairing for
          each
    """
    # step 1
    print('Doing Active Learning so actually collect {} clips and select the best 1/{} (put into pairs) using {} method.'.format(
        2 * args.selection_factor * num_labels_requested, 2 * args.selection_factor, args.active_method))
    print("Also, we're using the new clip pair acquisition method.")
    rand_clips, rand_rews = agent_experience.sample_singles(2 * args.selection_factor * num_labels_requested)
    # step 2
    sample_variance_per_clip = compute_sample_var_clip(rand_clips, reward_model, args)
    ref_clip_idx = np.argpartition(sample_variance_per_clip, 0)[0] # TODO this might become `ref_clips_idx` based on point 2 in the docstring. You'd just need to modify `0` to be `[:num_labels_requested]`
    assert ref_clip_idx.shape == ()
    ref_clip = rand_clips[ref_clip_idx]
    assert ref_clip.shape == (args.clip_length, args.obs_act_shape)
    repeated_ref_clip = np.repeat(np.expand_dims(ref_clip, axis=0), repeats=args.selection_factor*num_labels_requested, axis=0)
    assert repeated_ref_clip.shape == rand_clips.shape == (args.selection_factor*num_labels_requested, args.clip_length, args.obs_act_shape)
    rand_clips_paired_w_ref = np.stack((repeated_ref_clip, rand_clips), axis=1)
    assert rand_clips_paired_w_ref.shape == (args.selection_factor*num_labels_requested, 2, args.clip_length, args.obs_act_shape)

    # compute corresponding rews for use later on (need access to this in order to normalise rewards across prefs_buffer)
    ref_clip_rew = rand_rews[ref_clip_idx]
    repeated_ref_clip_rew = np.repeat(np.expand_dims(ref_clip_rew, axis=0), repeats=args.selection_factor*num_labels_requested, axis=0)
    rand_clips_paired_w_ref_rews = np.stack((repeated_ref_clip_rew, rand_rews), axis=1)
    assert rand_clips_paired_w_ref_rews.shape == (args.selection_factor*num_labels_requested, 2, args.clip_length)

    # step 3
    if args.active_method == 'BALD':
        info_per_clip_pair = compute_info_gain(rand_clips_paired_w_ref, reward_model, args)
    elif args.active_method == 'var_ratios':
        info_per_clip_pair = compute_var_ratio(rand_clips_paired_w_ref, reward_model, args)
    elif args.active_method == 'max_entropy':
        info_per_clip_pair = compute_pred_entropy(rand_clips_paired_w_ref, reward_model, args)
    elif args.active_method == 'naive_variance':
        info_per_clip_pair = compute_sample_var_clip_pair(rand_clips_paired_w_ref, reward_model, args)
    else:
        raise RuntimeError("You specified {} as the active_method type, but I don't know what that is!".format(args.active_method))
    # step 4
    idx = np.argpartition(info_per_clip_pair, -num_labels_requested)[-num_labels_requested:] # see: tinyurl.com/ya7xr4kn

    # compute mu
    returns = rand_clips_paired_w_ref_rews.sum(axis=-1) # sum up rews in each clip
    if args.force_label_choice:
        rand_mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                        np.where(returns[:, 0] == returns[:, 1], random.choice([0, 1]), 0))
    else: # allow clips to be labeled as 0.5
        rand_mus = np.where(returns[:, 0] > returns[:, 1], 1, 
                        np.where(returns[:, 0] == returns[:, 1], 0.5, 0))
    assert rand_mus.shape == (args.selection_factor*num_labels_requested,)

    clip_pairs, rews, mus = rand_clips_paired_w_ref[idx], rand_clips_paired_w_ref_rews[idx], rand_mus[idx] # returned indices are not sorted
    log_info_gain(info_per_clip_pair, idx, writer1, writer2, round_num=i_train_round)
    log_acquisitions(mus, rand_mus, rews, rand_rews, writer1, writer2, round_num=i_train_round)
    return clip_pairs, rews, mus


def sample_reward_model(reward_model, clips, args):
    """Takes array `clips` which must have shape[-1] == args.obs_act_shape
       and uses sampling method `args.uncert_method` to generate samples
       from the approximate poserior `reward_model`.
       Returns `r_preds_per_oa_pair` of the same shape as `clips` but with
       the obs_act_shape dimension removed.
    """
    batch_size = clips.shape[0] # TODO remove; this is only used for asserts
    clips_tensor = torch.from_numpy(clips).float()
    if args.uncert_method == 'MC':
        reward_model.train() # MC dropout
        r_preds_per_oa_pair = torch.cat([
            reward_model(clips_tensor).detach() for _ in range(args.num_MC_samples)
        ], dim=-1) # concatenate r_preds for same s-a pairs together
        check_num_samples = args.num_MC_samples
    elif args.uncert_method == 'ensemble':
        r_preds_per_oa_pair = reward_model.forward_all(clips_tensor).detach() # TODO check this line
        check_num_samples = reward_model.ensemble_size
    else:
        raise NotImplementedError("You specified {} as the `uncert_method`, but I don't know what that is!".format(args.uncert_method))
    assert r_preds_per_oa_pair.shape[0] == batch_size
    assert r_preds_per_oa_pair.shape[-2] == args.clip_length 
    assert r_preds_per_oa_pair.shape[-1] == check_num_samples
    if len(r_preds_per_oa_pair.shape) == 4:
        r_preds_per_oa_pair.shape[1] == 2
    return r_preds_per_oa_pair


def compute_sample_var_clip(rand_clips, reward_model, args):
    """Takes array `rand_clips` of shape (batch_size, clip_length, obs_act_shape),
       and computes `args.num_MC_samples` stochastic forward passes through `reward_model`.
       Returns an array of shape (batch_size,) of the variance of samples from
       `reward_model` approximate posterior, for each clip in `rand_clips`,
       where the sample variance of a clip is defined as the sum of the variance of
       each (obs, action) pair in the clip.
       Approximate posterior comes from either stochastic forward passes
       or by sampling from each reward predictor in the ensemble
       (depending on args.uncert_method).
    """
    r_preds_per_oa_pair = sample_reward_model(reward_model, rand_clips, args)
    batch_size = rand_clips.shape[0] # TODO remove; this is only used for asserts
    var_r_preds_per_oa_pair = r_preds_per_oa_pair.var(dim=-1) # take variance across r_preds for each s-a pair
    assert var_r_preds_per_oa_pair.shape == (batch_size, args.clip_length)
    var_r_preds_per_clip = var_r_preds_per_oa_pair.sum(dim=-1)
    assert var_r_preds_per_clip.shape == (batch_size,)
    return var_r_preds_per_clip.numpy()


def compute_sample_var_clip_pair(rand_clip_pairs, reward_model, args):
    """Takes np.array rand_clip_pairs with shape
       (batch_size, 2, clip_length, obs_act_shape)
       as well as reward_model, and returns np.array with shape
       (batch_size,) of the sample variance of the predictions
       according to `reward_model` of which clip in the pair
       is preferred by the annotator.
       Samples are drawn from `reward_model` approximate posterior
       which we get from either stochastic forward passes
       or by sampling from each reward predictor in the ensemble
       (depending on args.uncert_method).
    """
    r_preds_per_oa_pair = sample_reward_model(reward_model, rand_clip_pairs, args)
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    check_num_samples = r_preds_per_oa_pair.shape[-1]
    exp_sum_r_preds_per_batch_pair_draw = r_preds_per_oa_pair.sum(dim=2).exp()
    assert exp_sum_r_preds_per_batch_pair_draw.shape == (batch_size, 2, check_num_samples)
    p_hat_12_per_batch_draw = exp_sum_r_preds_per_batch_pair_draw[:, 0, :] / exp_sum_r_preds_per_batch_pair_draw.sum(dim=1)
    assert p_hat_12_per_batch_draw.shape == (batch_size, check_num_samples)
    Var_p_hat_12_per_batch = p_hat_12_per_batch_draw.var(dim=1)
    assert Var_p_hat_12_per_batch.shape == (batch_size,)
    return Var_p_hat_12_per_batch


def compute_info_gain(rand_clip_pairs, reward_model, args):
    """Takes np.array rand_clip_pairs with shape
       (batch_size, 2, clip_length, obs_act_shape)
       as well as reward_model, and returns np.array with shape
       (batch_size,) of the information gain
       resulting from adding each clip pair to the dataset, as per
       the BALD algo (Houlsby et al., 2011), or more
       specifically its approximation using samples from either the
       ensemble `reward_model` or using MC dropout (Gal et al. 2017)
    """
    r_preds_per_oa_pair = sample_reward_model(reward_model, rand_clip_pairs, args)
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    check_num_samples = r_preds_per_oa_pair.shape[-1]
    exp_sum_r_preds_per_batch_pair_draw = r_preds_per_oa_pair.sum(dim=2).exp()
    assert exp_sum_r_preds_per_batch_pair_draw.shape == (batch_size, 2, check_num_samples)
    p_hat_12_per_batch_draw = exp_sum_r_preds_per_batch_pair_draw[:, 0, :] / exp_sum_r_preds_per_batch_pair_draw.sum(dim=1)
    assert p_hat_12_per_batch_draw.shape == (batch_size, check_num_samples)
    E_p_hat_12_per_batch = p_hat_12_per_batch_draw.mean(dim=1)
    assert E_p_hat_12_per_batch.shape == (batch_size,)
    H_y_xD = F.binary_cross_entropy(input=E_p_hat_12_per_batch, target=E_p_hat_12_per_batch, reduction='none')

    # use the *same* draws from the posterior to approximate the second term
    X_entropy_per_batch_draw = F.binary_cross_entropy(input=p_hat_12_per_batch_draw, target=p_hat_12_per_batch_draw, reduction='none')
    assert X_entropy_per_batch_draw.shape == (batch_size, check_num_samples)
    E_H_y_xDw = X_entropy_per_batch_draw.mean(dim=1)
    assert E_H_y_xDw.shape == (batch_size,)

    info_gain = H_y_xD - E_H_y_xDw
    # assert (info_gain >= 0).all()
    return info_gain

def compute_pred_entropy(clip_pairs, reward_model, args):
    """Takes np.array `clip_pairs` with shape
       (batch_size, 2, clip_length, obs_act_shape)
       as well as `reward_model`, and returns np.array with shape
       (batch_size,) of the predictive entropy of each clip pair
       in the batch (Shannon, 1948).
       As per Yarin's Thesis, we approximate p(y=c | x, D_train)
       by averaging the T probability vectors from T stochastic
       forward passes (or rather, in our setting, the prob obtained
       from applying the ELO function to stochastic forward passes
       through `reward_model`). We perform the stochastic forward
       passes with either the ensemble `reward_model`
       or using MC dropout (Gal et al. 2017).
       Note that this is essentially the same acquisition function as
       compute_info_gain (BALD) except *without the second term*
    """
    r_preds_per_oa_pair = sample_reward_model(reward_model, clip_pairs, args)
    batch_size, _, clip_length, _ = clip_pairs.shape # used only in asserts
    check_num_samples = r_preds_per_oa_pair.shape[-1]
    exp_sum_r_preds_per_batch_pair_draw = r_preds_per_oa_pair.sum(dim=2).exp()
    assert exp_sum_r_preds_per_batch_pair_draw.shape == (batch_size, 2, check_num_samples)
    p_hat_12_per_batch_draw = exp_sum_r_preds_per_batch_pair_draw[:, 0, :] / exp_sum_r_preds_per_batch_pair_draw.sum(dim=1)
    assert p_hat_12_per_batch_draw.shape == (batch_size, check_num_samples)
    E_p_hat_12_per_batch = p_hat_12_per_batch_draw.mean(dim=1)
    assert E_p_hat_12_per_batch.shape == (batch_size,)
    return F.binary_cross_entropy(input=E_p_hat_12_per_batch, target=E_p_hat_12_per_batch, reduction='none')

def compute_var_ratio(clip_pairs, reward_model, args):
    """Takes np.array `clip_pairs` with shape
       (batch_size, 2, clip_length, obs_act_shape)
       as well as `reward_model`, and returns np.array with shape
       (batch_size,) of the variation ratio of each clip pair
       in the batch (Freeman, 1965).
       As per Deep Bayesian Active Learning with Image Data (2017),
       we approximate p(y=c | x, D_train)
       by averaging the T probability vectors from T stochastic
       forward passes (or rather, in our setting, the prob obtained
       from applying the ELO function to stochastic forward passes
       through `reward_model`). We perform the stochastic forward
       passes with either the ensemble `reward_model`
       or using MC dropout (Gal et al. 2017).
       NB there seems to be a discrepancy between the "mode" version
       version presented in Yarin's thesis (p.51) and the one
       implemented here and presented in the 2017 paper.
       But I'm sure I've misunderstood this, and that they are
       actually identical.
    """
    r_preds_per_oa_pair = sample_reward_model(reward_model, clip_pairs, args) # TODO there is a lot of copy-pasted code in these last 4 functions
    batch_size, _, clip_length, _ = clip_pairs.shape # used only in asserts
    check_num_samples = r_preds_per_oa_pair.shape[-1]
    exp_sum_r_preds_per_batch_pair_draw = r_preds_per_oa_pair.sum(dim=2).exp()
    assert exp_sum_r_preds_per_batch_pair_draw.shape == (batch_size, 2, check_num_samples)
    p_hat_12_per_batch_draw = exp_sum_r_preds_per_batch_pair_draw[:, 0, :] / exp_sum_r_preds_per_batch_pair_draw.sum(dim=1)
    assert p_hat_12_per_batch_draw.shape == (batch_size, check_num_samples)
    E_p_hat_12_per_batch = p_hat_12_per_batch_draw.mean(dim=1)
    assert E_p_hat_12_per_batch.shape == (batch_size,)
    max_p_12_p21 = np.maximum(E_p_hat_12_per_batch, 1 - E_p_hat_12_per_batch) # element-wise max of 2 arrays
    assert max_p_12_p21.shape == (batch_size,)
    return 1 - max_p_12_p21


def log_info_gain(info_per_clip_pair, idx, writer1, writer2, round_num):
    """1. Plots a bar chart with the info of each clip pair
          Clips pairs which were selected (given by idx) are orange; rest are blue.
       2. Plots two scalars: info summed over all clip pairs
                             info summed over selected clip pairs
       All plotting is done by logging to Tensorboard.
    """
    assert len(info_per_clip_pair.shape) == 1
    num_pairs = len(info_per_clip_pair)
    colours = ['tab:orange' if i in idx else 'tab:blue' for i in range(num_pairs)]
    info_bars = plt.figure()
    plt.title('Information gain per clip pair')
    plt.xlabel('Clip pairs')
    plt.ylabel('Metric of info gain')
    plt.bar(np.arange(num_pairs), info_per_clip_pair, color=colours)
    writer1.add_figure('info_gain_per_clip_pair', info_bars, round_num)

    total_info = info_per_clip_pair.sum()
    selected_info = info_per_clip_pair[idx].sum()
    # print('Total info: {}'.format(total_info))
    # print('Selected info: {}'.format(selected_info))
    writer1.add_scalar('5.info_gain_per_round_Total_blue_Selected_orange', selected_info, round_num)
    writer2.add_scalar('5.info_gain_per_round_Total_blue_Selected_orange', total_info, round_num)

def log_acquisitions(mus, rand_mus, rews, rand_rews, writer1, writer2, round_num):
    """Plots two histograms: the labels and return
       for each clip pair candidate and acquisition.
    """
    writer1.add_histogram('10.labels candidate and acquired', mus, round_num)
    writer2.add_histogram('10.labels candidate and acquired', rand_mus, round_num)
    writer1.add_histogram('11.return of clip pairs candidate and acquired', rews.sum(-1).sum(-1), round_num)
    writer2.add_histogram('11.return of clip pairs candidate and acquired', rand_rews.sum(-1).sum(-1), round_num)