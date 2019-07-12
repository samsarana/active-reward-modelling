"""Classes and functions to do reward learning"""

import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PrefsBuffer():
    def __init__(self, capacity, clip_shape):
        clip_length, obs_act_length = clip_shape
        self.clip_pairs = np.zeros(shape=(capacity, 2, clip_length, obs_act_length)) # 2 because preference is on clip *pair*
        self.rewards = np.zeros(shape=(capacity, 2, clip_length))
        self.mus = np.zeros(shape=capacity)
        self.capacity = capacity
        self.current_length = 0 # maintain the current length to help with sampling from the fixed size array
        self.clip_length = clip_length
        self.obs_act_length = obs_act_length

    def push(self, new_clip_pairs, new_rews, new_mus):
        """Takes
            new_clip_pairs.shape == (_, 2, clip_length, obs_act_length)
            new_mus.shape        == (_,)
            and pushes them onto the circular buffers self.clip_pairs
            and self.mus
        """
        len_new_pairs = len(new_clip_pairs)
        assert len_new_pairs == len(new_mus)
        self.clip_pairs = np.roll(self.clip_pairs, len_new_pairs, axis=0)
        self.rewards = np.roll(self.rewards, len_new_pairs, axis=0)
        self.mus = np.roll(self.mus, len_new_pairs)
        self.clip_pairs[:len_new_pairs] = new_clip_pairs
        self.rewards[:len_new_pairs] = new_rews
        self.mus[:len_new_pairs] = new_mus
        if self.current_length < self.capacity:
            self.current_length += len_new_pairs

    def sample(self, batch_size):
        """Returns *tensors* (clip_pair_batch, mu_batch) where
           clip_pair_batch.shape == (batch_size, 2, clip_len, obs_size+act_size)
           mu_batch.shape        == (batch_size, [1])
           clip_pairs and correspondings mus are sampled u.a.r. with replacement.
           In the flow of computation, data is numpy till here
           because it has useful roll and sampling abstractions.
           But from here on, everything will be tensors because we no longer need
           to push to buffers/index/sample, but rather do forward pass through NN
           and computation of loss (for which we need to track gradients)
        """
        idx = np.random.choice(self.current_length, size=batch_size)
        clip_pair_batch, mu_batch = self.clip_pairs[idx], self.mus[idx] # TODO fancy indexing is slow. is this a bottleneck?
        assert clip_pair_batch.shape == (batch_size, 2, self.clip_length, self.obs_act_length) # asinine assert statement
        assert mu_batch.shape == (batch_size, )
        return torch.from_numpy(clip_pair_batch).float(), torch.from_numpy(mu_batch).float()

    def all_flat_sa_pairs(self):
        """Returns float *tensor* of shape (-1, obs_act_length) with all
           the current (state, action) pairs in the buffer.
           Used to compute the mean and variance of reward model across
           all examples in the prefs_buffer (in order to normalise rewards
           sent to agent)
        """
        current_sa_pairs = self.clip_pairs[:self.current_length]
        flat_sa_pairs = current_sa_pairs.reshape((-1, self.obs_act_length))
        return torch.from_numpy(flat_sa_pairs).float()

    def compute_mean_var_GT(self):
        """Since we stored true rewards for each state-action pair in the
           prefs_buffer, we can easily return mean and var of ground-truth
           reward function over the examples in the buffer.
           We will use this for plotting purposes, to compare *normalised*
           true and modelled rewards.
        """
        all_rewards_flat = self.rewards[:self.current_length].reshape(-1)
        return all_rewards_flat.mean(), all_rewards_flat.var()


class RewardModel(nn.Module):
    """Parameterises r_hat : states x actions -> real rewards
       Approximation of true reward, trained by supervised learning
       on preferences over trajectory segments as in Christiano et al. 2017
    """
    def __init__(self, state_size, action_size, args):
        """Feedforward NN with 2 hidden layers"""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size + action_size, args.hid_units_rm),
            nn.ReLU(),
            nn.Dropout(args.p_dropout_rm),
            nn.Linear(args.hid_units_rm, args.hid_units_rm),
            nn.ReLU(),
            nn.Dropout(args.p_dropout_rm),
            nn.Linear(args.hid_units_rm, 1),
        )
        
    def forward(self, x):
        return self.layers(x)


class RewardModelEnsemble(nn.Module):
    """Parameterises r_hat : states x actions -> real rewards
       Approximation of true reward, trained by supervised learning
       on preferences over trajectory segments as in Christiano et al. 2017
       Ouput is an average of `args.size_rm_ensemble` networks
       TODO check this is how Christiano actually implements it.
       In particular, do they say something about normalising
       the output from each network in the ensemble separately?
       Is this important?
    """
    def __init__(self, state_size, action_size, args):
        """Feedforward NN with 2 hidden layers"""
        super().__init__()
        self.ensemble_size = args.size_rm_ensemble
        assert self.ensemble_size >= 2
        for ensemble_num in range(self.ensemble_size):
            setattr(self, 'layers{}'.format(ensemble_num), 
                    nn.Sequential(
                        nn.Linear(state_size + action_size, args.hid_units_rm),
                        nn.ReLU(),
                        nn.Dropout(args.p_dropout_rm),
                        nn.Linear(args.hid_units_rm, args.hid_units_rm),
                        nn.ReLU(),
                        nn.Dropout(args.p_dropout_rm),
                        nn.Linear(args.hid_units_rm, 1),
                        )
                    )
        
    def forward(self, x):
        output = 0
        for ensemble_num in range(self.ensemble_size):
            net = getattr(self, 'layers{}'.format(ensemble_num))
            output += net(x)
        return output / self.ensemble_size

    def forward_all(self, x):
        """Instead of averaging output across `ensemble_size`
           networks, return tensor of the output from each network
           in ensemble. Results from different ensembles are
           concatenated on the innermost dimension.
           This will be used, for example, in the BALD algo.
        """
        outputs = []
        for ensemble_num in range(self.ensemble_size):
            net = getattr(self, 'layers{}'.format(ensemble_num))
            outputs.append(net(x))
        return torch.cat(outputs, dim=-1)

    def variance(self, x):
        """Returns predictive variance of the networks
           in the ensemble, on input x
        """
        batch_size, _, clip_length, _ = x.shape # only used for assert
        outputs = []
        for ensemble_num in range(self.ensemble_size):
            net = getattr(self, 'layers{}'.format(ensemble_num))
            outputs.append(net(x))
        outputs_tensor = torch.cat(outputs, dim=-1)
        assert outputs_tensor.shape == (batch_size, 2, clip_length, self.ensemble_size)
        return outputs_tensor.var(dim=-1)

    # def forward_no_ave(self, x):
    #     """Instead of averaging output across 3 networks, return
    #        a 3-tuple of the output from each network in ensemble
    #        Do not use for learning, only for prediction!
    #     """
    #     return self.layers0(x), self.layers1(x), self.layers2(x)


def compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, obs_shape, act_shape):
    """Implementation of the loss function on p.5 of Christiano et al.
        Assumes that human's probability of preferring a segment sigma_i
        depends exponentially on the value of r_hat, which we view as
        the 'latent' reward, summed over the length of each segment
        Takes args, obs_shape, act_shape just to do the assert dimension checks
    """
    assert not torch.isnan(r_hats_batch.detach()).numpy().any()
    assert r_hats_batch.shape == (args.batch_size_rm, 2, args.clip_length)
    exp_sum_rhb = r_hats_batch.sum(dim=2).exp() # rhb: r_hats_batch. sum is over clip_length
    assert exp_sum_rhb.shape == (args.batch_size_rm, 2)
    exp_sum_rhb_1 = exp_sum_rhb[:, 0]
    assert exp_sum_rhb_1.shape == (args.batch_size_rm,)
    exp_sum_rhb_sum = exp_sum_rhb.sum(dim=1) # sum is over pair-dimension
    # reminder: 0=batch, 1=pair, 2=clip_length (now gone), 3=oa_pair (now gone)
    assert exp_sum_rhb_sum.shape == (args.batch_size_rm,)
    p_hat_12_batch = exp_sum_rhb_1 / exp_sum_rhb_sum
    assert p_hat_12_batch.shape == (args.batch_size_rm, )
    assert mu_batch.shape == (args.batch_size_rm, )
    loss_rm = F.binary_cross_entropy(input=p_hat_12_batch, target=mu_batch, reduction='sum')
    # loss_batch = mu_batch * [log]p_hat_12_batch + (1 - mu_batch) * [log](1 - p_hat_12_batch)
    # loss_rm = [-]loss_batch.sum() # 2 errors on these 2 lines! (forgot to make -ve and take take logs of probs)
    assert loss_rm.shape == ()
    assert not torch.isnan(loss_rm)
    return loss_rm


def compute_loss_rm(r_hats_batch, mu_batch):
    """Clean, assert-free version of the above
    """
    exp_sum_r_hats_batch = r_hats_batch.sum(dim=2).exp()
    p_hat_12_batch = exp_sum_r_hats_batch[:, 0] / exp_sum_r_hats_batch.sum(dim=1)
    return F.binary_cross_entropy(input=p_hat_12_batch, target=mu_batch, reduction='sum')


def eval_rm_correlation(reward_model, env, agent, args, obs_shape, act_shape, rollout_steps, num_rollouts):
    """1. Computes `r_xy`, the Pearson product-moment correlation between
          true and predicted rewards, accumulated over `rollout_steps` * `num_rollouts`.
          The set of state-action pairs used to compute the correlation is obtained
          from rollouts of the `agent` in the `env` using an epsilon-greedy policy,
          with constant epsilon=0.05.
          Rollouts are of length `rollout_steps`. Since `env` is assumed to have
          never-ending episodes, this is simple to implement.
       2. Creates the corresponding `scatter_plot` of true v predicted reward, on this
          same dataset. `scatter_plot` generated using matplotlib.
       3. `scatter_plot_norm` is the same, except that it uses normalised rewards.
          To do so, we compute mean and variance across the `sa_pairs` collected.
          NB these are different mean and variance to the ones used to normalise rewards
          sent to agent in the main training loop (there, is it across the Preferences
          Buffer, as per Ibarz). I chose to use sa_pairs collected here because
          it's easier -- I don't have a principled reason to do so.
       4. `scatter_plot_norm_error_bars` also includes error bars on the 10th and 90th
          percentiles of samples drawn from the reward model using MC-Dropout with p=0.2
          [experimental]
       Returns: r_xy, dict of scatter plots 2-4
    """
    # 0. generate data
    sa_pairs = torch.zeros(size=(num_rollouts*rollout_steps, obs_shape+act_shape))
    r_true = np.zeros(shape=(num_rollouts * rollout_steps))
    step = 0
    for _ in range(num_rollouts):
        state = env.reset()
        for _ in range(rollout_steps):
            action = agent.act(state, epsilon=0.05)
            assert env.action_space.contains(action)
            next_state, reward, _, _ = env.step(action)
            # record step information
            sa_pair = torch.FloatTensor(np.append(state, action))
            sa_pairs[step] = sa_pair
            r_true[step] = reward
            state = next_state
            step += 1
    
    # 1. compute r_xy
    reward_model.eval() # dropout off
    r_pred = reward_model(sa_pairs).detach().squeeze().numpy()
    assert r_true.shape == (num_rollouts * rollout_steps,)
    assert r_pred.shape == (num_rollouts * rollout_steps,)
    r_xy = np.corrcoef(r_true, r_pred)[0][1]

    # 2. scatter plot
    scatter_plots = {}
    scatter_plots['non-norm'] = plt.figure(figsize=(15,8))
    plt.title('Correlation of predicted and true reward. r_xy = {:.2f}\nAccumulated over {} steps'.format(
        r_xy, args.corr_rollout_steps * args.corr_num_rollouts))
    plt.xlabel('True reward')
    plt.ylabel('Predicted reward')
    plt.scatter(r_true, r_pred)

    # 3. normalised
    r_true_norm = (r_true - r_true.mean()) / np.sqrt(r_true.var() + 1e-8)
    r_pred_norm = (r_pred - r_pred.mean()) / np.sqrt(r_pred.var() + 1e-8)
    scatter_plots['norm'] = plt.figure(figsize=(15,8))
    plt.title('Correlation of (normalised) predicted and true reward. r_xy = {:.2f}\nAccumulated over {} steps'.format(
        r_xy, args.corr_rollout_steps * args.corr_num_rollouts))
    plt.xlabel('True reward (normalised)')
    plt.ylabel('Predicted reward (normalised)')
    plt.scatter(r_true_norm, r_pred_norm)

    # 4. error bars
    reward_model.train() # we'll now draw samples from posterior to get error bars on scatter plot
    r_pred_samples = np.array([reward_model(sa_pairs).detach().squeeze().numpy() for _ in range(100)])
    # dims: 0=samples, 1=examples

    r_pred_samples_norm = (r_pred_samples - r_pred.mean()) / np.sqrt(r_pred.var() + 1e-8)
    tenth, ninetieth = np.percentile(r_pred_samples_norm, 10, axis=0), np.percentile(r_pred_samples_norm, 90, axis=0)
    assert tenth.shape == (num_rollouts * rollout_steps,)
    assert ninetieth.shape == (num_rollouts * rollout_steps,)
    minus_err = r_pred_norm - tenth
    assert (minus_err >= 0).all()
    plus_err = ninetieth - r_pred_norm

    scatter_plots['norm_error_bars'] = plt.figure(figsize=(15,8))
    plt.title('Correlation of (normalised) predicted and true reward with error bars. r_xy = {:.2f}\nAccumulated over {} steps'.format(
        r_xy, args.corr_rollout_steps * args.corr_num_rollouts))
    # maybe do something later about changing colour of reward and error bars corresp. to out-of-distribution state-action pairs
    plt.errorbar(r_true_norm, r_pred_norm, yerr=np.array([plus_err, minus_err]), fmt='o', ecolor='g', alpha=0.3, capsize=5) # would be nicer to make errors(_norm) a namedtuple (or namedlist if plt.errorbar yerr arg needs a list), s.t. it can be passed straight in, here
    plt.xlabel('True reward (normalised)')
    plt.ylabel('Modelled reward (normalised)')

    return r_xy, scatter_plots


def log_correlation(r_xy, plots, writer, round_num):
    writer.add_scalar('4.r_xy', r_xy, round_num)
    writer.add_figure('4.alignment/1-non-norm', plots['non-norm'], round_num) # we can have mutliple figures with the same tag and scroll through them!
    writer.add_figure('4.alignment/2-norm', plots['norm'], round_num)
    writer.add_figure('4.alignment/3-norm_error_bars', plots['norm_error_bars'], round_num)