import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from reward_model import RewardModelEnsemble

def compute_entropy_reductions(rand_clip_pairs, reward_model, num_MC_samples=100):
    """Takes np.array rand_clip_pairs with shape
       (batch_size, 2, clip_length, obs_act_length)
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
    assert torch.all(torch.lt(torch.abs(torch.add(H_y_xD, -check)), 1e-6))

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
    assert torch.all(torch.lt(torch.abs(torch.add(E_H_y_xDw, -check2)), 1e-6))

    info_gain = H_y_xD - E_H_y_xDw
    assert (info_gain >= 0).all()
    return info_gain


def  compute_MC_variance(rand_clip_pairs, reward_model, num_MC_samples=100):
    """Takes np.array rand_clip_pairs with shape
       (batch_size, 2, clip_length, obs_act_length)
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
       (batch_size, 2, clip_length, obs_act_length)
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