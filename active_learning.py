"""Functions to do Active Learning"""

import random, logging
import numpy as np
import torch
import torch.nn.functional as F

def acq_random(rand_clip_pairs, num_labels, reward_model, args):
    n_clip_pairs, _, _, _ = rand_clip_pairs.shape
    idx = np.random.choice(n_clip_pairs, size=num_labels, replace=False)
    info_per_clip_pair = None
    return idx, info_per_clip_pair


def acq_BALD(rand_clip_pairs, num_labels, args, reward_model):
    info_per_clip_pair = compute_info_gain(rand_clip_pairs, reward_model, args)
    idx = np.argpartition(info_per_clip_pair, -num_labels)[-num_labels:].numpy() # see: tinyurl.com/ya7xr4kn
    return idx, info_per_clip_pair

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
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    p_hat_12_per_batch_draw = sample_p_hat_12_per_batch(reward_model, rand_clip_pairs, args)
    check_num_samples = p_hat_12_per_batch_draw.shape[-1] # used only in asserts
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


def acq_mean_std(rand_clip_pairs, num_labels, args, reward_model):
    info_per_clip_pair = compute_sample_var_clip_pair(rand_clip_pairs, reward_model, args)
    idx = np.argpartition(info_per_clip_pair, -num_labels)[-num_labels:].numpy() # see: tinyurl.com/ya7xr4kn
    return idx, info_per_clip_pair

def compute_sample_var_clip_pair(rand_clip_pairs, reward_model, args): # TODO pairwise combine the next 8 functions?
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
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    p_hat_12_per_batch_draw = sample_p_hat_12_per_batch(reward_model, rand_clip_pairs, args)
    Var_p_hat_12_per_batch = p_hat_12_per_batch_draw.var(dim=1)
    assert Var_p_hat_12_per_batch.shape == (batch_size,)
    return Var_p_hat_12_per_batch


def acq_max_entropy(rand_clip_pairs, num_labels, args, reward_model):
    info_per_clip_pair = compute_pred_entropy(rand_clip_pairs, reward_model, args)
    idx = np.argpartition(info_per_clip_pair, -num_labels)[-num_labels:].numpy() # see: tinyurl.com/ya7xr4kn
    return idx, info_per_clip_pair

def compute_pred_entropy(rand_clip_pairs, reward_model, args):
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
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    p_hat_12_per_batch_draw = sample_p_hat_12_per_batch(reward_model, rand_clip_pairs, args)
    E_p_hat_12_per_batch = p_hat_12_per_batch_draw.mean(dim=1)
    assert E_p_hat_12_per_batch.shape == (batch_size,)
    return F.binary_cross_entropy(input=E_p_hat_12_per_batch, target=E_p_hat_12_per_batch, reduction='none')


def acq_var_ratios(rand_clip_pairs, num_labels, args, reward_model):
    info_per_clip_pair = compute_var_ratio(rand_clip_pairs, reward_model, args)
    idx = np.argpartition(info_per_clip_pair, -num_labels)[-num_labels:].numpy() # see: tinyurl.com/ya7xr4kn
    return idx, info_per_clip_pair

def compute_var_ratio(rand_clip_pairs, reward_model, args):
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
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    p_hat_12_per_batch_draw = sample_p_hat_12_per_batch(reward_model, rand_clip_pairs, args)
    E_p_hat_12_per_batch = p_hat_12_per_batch_draw.mean(dim=1)
    assert E_p_hat_12_per_batch.shape == (batch_size,)
    max_p_12_p21 = np.maximum(E_p_hat_12_per_batch, 1 - E_p_hat_12_per_batch) # element-wise max of 2 arrays
    assert max_p_12_p21.shape == (batch_size,)
    return 1 - max_p_12_p21


def sample_p_hat_12_per_batch(reward_model, rand_clip_pairs, args):
    """Takes `reward_model` mapping state-action pairs to predicted rewards
       and `rand_clip_pairs` of shape (batch_size, 2, clip_length, obs_act_length)
       and computes p_hat_12 as defined in Eq. (1) of Christiano et al. (2017).
       Returns `p_hat_12_per_batch_draw` which is just this quantity, but recomputed
       several times using different draws from `reward_model` approx posterior.
       (The method for sampling from approx posterior is given in args.uncert_method).
       p_hat_12_per_batch_draw.shape == (batch_size, n_samples_from_approx_poserior)
    """
    r_preds_per_oa_pair = sample_reward_model(reward_model, rand_clip_pairs, args)
    batch_size, _, clip_length, _ = rand_clip_pairs.shape # used only in asserts
    check_num_samples = r_preds_per_oa_pair.shape[-1] # used only in asserts
    exp_sum_r_preds_per_batch_pair_draw = r_preds_per_oa_pair.sum(dim=2).exp()
    assert exp_sum_r_preds_per_batch_pair_draw.shape == (batch_size, 2, check_num_samples)
    p_hat_12_per_batch_draw = exp_sum_r_preds_per_batch_pair_draw[:, 0, :] / exp_sum_r_preds_per_batch_pair_draw.sum(dim=1)
    assert p_hat_12_per_batch_draw.shape == (batch_size, check_num_samples)
    return p_hat_12_per_batch_draw


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
            reward_model(clips_tensor, mode='batch').detach() for _ in range(args.num_MC_samples)
        ], dim=-1) # concatenate r_preds for same s-a pairs together
        check_num_samples = args.num_MC_samples
    elif args.uncert_method == 'ensemble':
        reward_model.eval() # dropout off at 'test' time i.e. when each model in ensemble to get uncertainty estimate. TODO check if this is correct
        r_preds_per_oa_pair = reward_model.forward_all(clips_tensor, mode='batch').detach() # TODO check this line
        check_num_samples = reward_model.ensemble_size
    else:
        raise NotImplementedError("You specified {} as the `uncert_method`, but I don't know what that is!".format(args.uncert_method))
    assert r_preds_per_oa_pair.shape[0] == batch_size
    assert r_preds_per_oa_pair.shape[-2] == args.clip_length 
    assert r_preds_per_oa_pair.shape[-1] == check_num_samples
    if len(r_preds_per_oa_pair.shape) == 4:
        r_preds_per_oa_pair.shape[1] == 2
    return r_preds_per_oa_pair.double() # all entropy calculations now done to double precision (Andreas' advice)