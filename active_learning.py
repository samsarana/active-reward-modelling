import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from reward_model import RewardModelEnsemble

def acquire_clip_pairs_v0(agent_experience, reward_model, num_labels_requested, args, writer1, writer2, i_train_round):
    print('Doing Active Learning, so actually collect {} labels and select the best 1/{} using {} method'.format(
        args.selection_factor * num_labels_requested, args.selection_factor, args.active_learning))
    rand_clip_pairs, rand_rews, rand_mus = agent_experience.sample_pairs(args.selection_factor * num_labels_requested)
    if args.active_learning == 'BALD-MC':
        info_per_clip_pair = compute_info_gain_MC(rand_clip_pairs, reward_model, args.num_MC_samples)
    elif args.active_learning == 'MC_variance':
        info_per_clip_pair = compute_MC_variance(rand_clip_pairs, reward_model, args.num_MC_samples)
    elif args.active_learning == 'ensemble_variance':
        info_per_clip_pair = compute_ensemble_variance(rand_clip_pairs, reward_model)
    else:
        raise RuntimeError("You specified {} as the active_learning type, but I don't know what that is!".format(args.active_learning))
    idx = np.argpartition(info_per_clip_pair, -num_labels_requested)[-num_labels_requested:] # see: tinyurl.com/ya7xr4kn
    clip_pairs, rews, mus = rand_clip_pairs[idx], rand_rews[idx], rand_mus[idx] # returned indices are not sorted
    log_active_learning(info_per_clip_pair, idx, writer1, writer2, round_num=i_train_round)
    return clip_pairs, rews, mus

def compute_sample_variance_clipwise(rand_clips, reward_model, args):
    """Takes array `rand_clips` of shape (batch_size, clip_length, obs_act_shape),
       and computes `args.num_MC_samples` stochastic forward passes through `reward_model`.
       Returns an array of shape (batch_size,) of the sample variance of the stochastic
       forward passes for each clip in `rand_clips`, where the variance of a clip
       is defined as the sum of the variance of each (obs, action) pair in the clip.
       TODO correct this docstring to mention either doing stochastic forward passes
       or ensemble method
    """
    batch_size = rand_clips.shape[0] # TODO remove; this is only used for asserts
    if args.active_learning in ['BALD-ensemble', 'ensemble_variance']: # TODO better: if args.uncert_method == 'ensemble'
        raise NotImplementedError
    else: # TODO elif args.uncert_method == 'MC'
        reward_model.train() # MC dropout
        clip_pairs_tensor = torch.from_numpy(rand_clips).float()
        r_preds_per_oa_pair = torch.cat([
            reward_model(clip_pairs_tensor).detach() for _ in range(args.num_MC_samples)
        ], dim=-1) # concatenate r_preds for same s-a pairs together
        assert r_preds_per_oa_pair.shape == (batch_size, args.clip_length, args.num_MC_samples)
        var_r_preds_per_oa_pair = r_preds_per_oa_pair.var(dim=-1) # take variance across r_preds for each s-a pair
        assert var_r_preds_per_oa_pair.shape == (batch_size, args.clip_length)
        var_r_preds_per_clip = var_r_preds_per_oa_pair.sum(dim=-1)
        assert var_r_preds_per_clip.shape == (batch_size,)
    return var_r_preds_per_clip.numpy()


def acquire_clip_pairs_v1(agent_experience, reward_model, num_labels_requested, args, writer1, writer2, i_train_round):
    """1. Samples m = `args.selection_factor * num_labels_requested` clips from agent_experience.
       2. Finds the clip with minimum uncertainty according to current reward_model (using the sample
           variance of multiple MC samples. See Yarin's thesis p.49, 51 for why this is justified.
           TODO 1. check that I am allowed to ignore the precision term when eval'ing uncertainty of 
           different x under the same model). This will be the reference clip `sigma_1`.
       3. For every other clip `sigma_2` in agent_experience, computes the mutual information
          between the predicted label of (`sigma_1`, `sigma_2`) and `reward_model` parameters.
          (This is for BALD. If `args.active_learning != 'bald'` then use some other method.)
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
    print('Doing Active Learning so actually collect {} labels and select the best 1/{} using {} method.'.format(
        args.selection_factor * num_labels_requested, args.selection_factor, args.active_learning))
    print("Also, we're using the new clip pair acquisition method.")
    rand_clips, rand_rews = agent_experience.sample_singles(args.selection_factor * num_labels_requested)
    # step 2
    sample_variance_per_clip = compute_sample_variance_clipwise(rand_clips, reward_model, args) # TODO we might also want to compute variance across ensemble instead of using MC-dropout
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

    # step 3 # TODO clean up ugly if-elifs. Also the taxonomy of methods is: ensemble/MC x BALD/var_ratios/max_entropy/naive variance. Structure code to reflect this.
    # in other words, we should have args.uncertainty_method and args.active_method
    if args.active_learning == 'BALD-MC':
        info_per_clip_pair = compute_info_gain_MC(rand_clips_paired_w_ref, reward_model, args.num_MC_samples)
    elif args.active_learning == 'BALD-ensemble':
        raise NotImplementedError
    elif args.active_learning == 'var_ratios':
        raise NotImplementedError
    elif args.active_learning == 'max_entropy':
        raise NotImplementedError
    elif args.active_learning == 'MC_variance':
        info_per_clip_pair = compute_MC_variance(rand_clips_paired_w_ref, reward_model, args.num_MC_samples)
    elif args.active_learning == 'ensemble_variance':
        info_per_clip_pair = compute_ensemble_variance(rand_clips_paired_w_ref, reward_model)
    else:
        raise RuntimeError("You specified {} as the active_learning type, but I don't know what that is!".format(args.active_learning))
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
    log_active_learning(info_per_clip_pair, idx, writer1, writer2, round_num=i_train_round)
    return clip_pairs, rews, mus


def compute_info_gain_MC(rand_clip_pairs, reward_model, num_MC_samples):
    """NB this is just an assert-free version of compute_entropy_reductions...
       Takes np.array rand_clip_pairs with shape
       (batch_size, 2, clip_length, obs_act_shape)
       as well as reward_model, and returns np.array with shape
       (batch_size,) of the information gain
       resulting from adding each clip pair to the dataset, as per
       the BALD algo (Houlsby et al., 2011), or more
       specifically its approximation using MC samples from
       (Gal et al. 2017)
    """
    # batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    reward_model.train() # MC-Dropout
    clip_pairs_tensor = torch.from_numpy(rand_clip_pairs).float()
    r_preds_per_oa_pair = torch.cat([
        reward_model(clip_pairs_tensor).detach() for _ in range(num_MC_samples)
    ], dim=-1) # concatenate r_preds for same s-a pairs together
    # assert r_preds_per_oa_pair.shape == (batch_size, 2, clip_length, num_MC_samples)
    exp_sum_r_preds_per_batch_pair_draw = r_preds_per_oa_pair.sum(dim=2).exp()
    # assert exp_sum_r_preds_per_batch_pair_draw.shape == (batch_size, 2, num_MC_samples)
    p_hat_12_per_batch_draw = exp_sum_r_preds_per_batch_pair_draw[:, 0, :] / exp_sum_r_preds_per_batch_pair_draw.sum(dim=1)
    # assert p_hat_12_per_batch_draw.shape == (batch_size, num_MC_samples)
    E_p_hat_12_per_batch = p_hat_12_per_batch_draw.mean(dim=1)
    # assert E_p_hat_12_per_batch.shape == (batch_size,)
    H_y_xD = F.binary_cross_entropy(input=E_p_hat_12_per_batch, target=E_p_hat_12_per_batch, reduction='none')

    # use the *same* draws from the posterior to approximate the second term
    X_entropy_per_batch_draw = F.binary_cross_entropy(input=p_hat_12_per_batch_draw, target=p_hat_12_per_batch_draw, reduction='none')
    # assert X_entropy_per_batch_draw.shape == (batch_size, num_MC_samples)
    E_H_y_xDw = X_entropy_per_batch_draw.mean(dim=1)
    # assert E_H_y_xDw.shape == (batch_size,)

    info_gain = H_y_xD - E_H_y_xDw
    # assert (info_gain >= 0).all()
    return info_gain

def compute_info_gain_ensemble(rand_clip_pairs, reward_model, num_MC_samples):
    pass

def compute_info_gain_MC_w_checks(rand_clip_pairs, reward_model, num_MC_samples):
    """Takes np.array rand_clip_pairs with shape
       (batch_size, 2, clip_length, obs_act_shape)
       as well as reward_model, and returns np.array with shape
       (batch_size,) of the entropy reduction (information gain)
       resulting from adding each clip pair to the dataset.
    """
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    reward_model.train() # MC-Dropout
    clip_pairs_tensor = torch.from_numpy(rand_clip_pairs).float()
    r_preds_per_oa_pair = torch.cat([
        reward_model(clip_pairs_tensor).detach() for _ in range(num_MC_samples)
    ], dim=-1) # concatenate r_preds for same s-a pairs together
    assert r_preds_per_oa_pair.shape == (batch_size, 2, clip_length, num_MC_samples)
    exp_sum_r_preds_per_batch_pair_draw = r_preds_per_oa_pair.sum(dim=2).exp()
    assert exp_sum_r_preds_per_batch_pair_draw.shape == (batch_size, 2, num_MC_samples)
    p_hat_12_per_batch_draw = exp_sum_r_preds_per_batch_pair_draw[:, 0, :] / exp_sum_r_preds_per_batch_pair_draw.sum(dim=1)
    assert p_hat_12_per_batch_draw.shape == (batch_size, num_MC_samples)
    E_p_hat_12_per_batch = p_hat_12_per_batch_draw.mean(dim=1)
    assert E_p_hat_12_per_batch.shape == (batch_size,)
    H_y_xD = F.binary_cross_entropy(input=E_p_hat_12_per_batch, target=E_p_hat_12_per_batch, reduction='none')

    # TODO remove check
    check = - (E_p_hat_12_per_batch       * E_p_hat_12_per_batch.log() + 
               (1 - E_p_hat_12_per_batch) * (1 - E_p_hat_12_per_batch).log()  )
    assert torch.all(torch.lt(torch.abs(torch.add(H_y_xD, -check)), 1e-4))

    # TODO is it correct NOT to do fresh draw from posterior?
    # NB when I tried doing fresh draws, this made (info_gain >= 0).all() False !! (which is bad...)
    # If you do want fresh draws after all, here is the code:
    # -------------------------------------------------------
    # r_preds_per_oa_pair = torch.cat([
    #     reward_model(clip_pairs_tensor).detach() for _ in range(num_MC_samples)
    # ], dim=-1) # concatenate r_preds for same s-a pairs together
    # exp_sum_r_preds_per_batch_pair_draw = r_preds_per_oa_pair.sum(dim=2).exp()
    # p_hat_12_per_batch_draw = exp_sum_r_preds_per_batch_pair_draw[:, 0, :] / exp_sum_r_preds_per_batch_pair_draw.sum(dim=1)
    # -------------------------------------------------------
    X_entropy_per_batch_draw = F.binary_cross_entropy(input=p_hat_12_per_batch_draw, target=p_hat_12_per_batch_draw, reduction='none')
    assert X_entropy_per_batch_draw.shape == (batch_size, num_MC_samples)
    E_H_y_xDw = X_entropy_per_batch_draw.mean(dim=1)
    assert E_H_y_xDw.shape == (batch_size,)

    # TODO remove check
    check1 = - (p_hat_12_per_batch_draw       * p_hat_12_per_batch_draw.log() + 
               (1 - p_hat_12_per_batch_draw) * (1 - p_hat_12_per_batch_draw).log()  )
    assert check1.shape == (batch_size, num_MC_samples)
    check2 = check1.mean(dim=1)
    assert check2.shape == (batch_size,)
    assert torch.all(torch.lt(torch.abs(torch.add(E_H_y_xDw, -check2)), 1e-4))

    info_gain = H_y_xD - E_H_y_xDw
    assert (info_gain >= 0).all()
    return info_gain


def compute_MC_variance(rand_clip_pairs, reward_model, num_MC_samples):
    """Takes np.array rand_clip_pairs with shape
       (batch_size, 2, clip_length, obs_act_shape)
       as well as reward_model, and returns np.array with shape
       (batch_size,) of the predictive variance of the reward model
       summed over all (state, action) pairs in the pair of clips
       (e.g. 50 state-action pairs with default settings)
       for each clip_pair, according to MC-Dropout.
    """
    reward_model.train() # MC-Dropout
    batch_size, _, clip_length, _ = rand_clip_pairs.shape
    clip_pairs_tensor = torch.from_numpy(rand_clip_pairs).float()
    r_preds_per_oa_pair = torch.cat([
        reward_model(clip_pairs_tensor).detach() for _ in range(num_MC_samples)
    ], dim=-1) # concatenate r_preds for same s-a pairs together
    assert r_preds_per_oa_pair.shape == (batch_size, 2, clip_length, num_MC_samples)
    var_r_preds_per_oa_pair = r_preds_per_oa_pair.var(dim=-1) # take variance across r_preds for each s-a pair
    assert var_r_preds_per_oa_pair.shape == (batch_size, 2, clip_length)
    var_r_preds_per_clip_pair = var_r_preds_per_oa_pair.sum(dim=-1).sum(dim=-1)
    assert var_r_preds_per_clip_pair.shape == (batch_size,)
    return var_r_preds_per_clip_pair.numpy()


def compute_ensemble_variance(rand_clip_pairs, reward_model):
    """Takes np.array rand_clip_pairs with shape
       (batch_size, 2, clip_length, obs_act_shape)
       as well as reward_model which must be an ensemble
       (in particular, it must have a variance(.) method)
       and returns np.array with shape
       (batch_size,) of the predictive variance of the ensemble
       summed over all (state, action) pairs in the pair of clips
       (e.g. 50 state-action pairs with default settings)
       for each clip_pair.
    """
    assert isinstance(reward_model, RewardModelEnsemble)
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # this line only used for asserts
    reward_model.eval() # no dropout
    clip_pairs_tensor = torch.from_numpy(rand_clip_pairs).float()
    pred_vars_per_oa_pair = reward_model.variance(clip_pairs_tensor).detach()
    assert pred_vars_per_oa_pair.shape == (batch_size, 2, clip_length)
    pred_vars_per_clip_pair = pred_vars_per_oa_pair.sum(dim=-1).sum(dim=-1)
    assert pred_vars_per_clip_pair.shape == (batch_size,)
    return pred_vars_per_clip_pair.numpy()


def log_active_learning(info_per_clip_pair, idx, writer1, writer2, round_num):
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