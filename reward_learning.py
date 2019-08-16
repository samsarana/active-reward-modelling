"""Classes and functions to do reward learning"""

import math, random, logging
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def train_reward_model(reward_model, prefs_buffer, optimizer_rm, args, writers, i_label):
    writer1, writer2 = writers
    epochs = args.n_epochs_pretrain_rm if i_label <= -1 else args.n_epochs_train_rm
    logging.info("Training reward model for {} epochs".format(epochs))
    reward_model.train() # dropout on
    # logging.info("reward_model weight before train {}: {}".format(i_label, list(reward_model.parameters())[0][0][0]))
    for epoch in range(epochs):
        with torch.autograd.detect_anomaly():
            clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
            if args.normalise_rm_while_training:
                raise RuntimeError("Normalising reward model while training is an untested idea. Why are you doing it?")
                # reward_model = compute_mean_var(reward_model, prefs_buffer) # TODO uncertain about the speed and correctness of recomputing mean/var every gradient update
                # r_hats_batch = reward_model(clip_pair_batch, normalise=True).squeeze(-1) # squeeze the oa_pair dimension that was passed through reward_model
                # loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
            else:
                if isinstance(reward_model, RewardModelEnsemble):
                    r_hats_batch_draw = reward_model.forward_all(clip_pair_batch, normalise=False).squeeze(-1)
                    loss_rm = compute_loss_rm_ensemble(r_hats_batch_draw, mu_batch)
                else:
                    assert isinstance(reward_model, RewardModel)
                    r_hats_batch = reward_model(clip_pair_batch, normalise=False).squeeze(-1)
                    loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
            # assert clip_pair_batch.shape == (args.batch_size_rm, 2, args.clip_length, args.obs_act_shape)
            # loss_rm = compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, args.obs_shape, args.act_shape)
            optimizer_rm.zero_grad()
            loss_rm.backward()
            optimizer_rm.step()
            writer1.add_scalar('6.reward_model_loss/label_{}'.format(i_label), loss_rm, epoch)
            # compute lower bound for loss_rm and plot this too. TODO check this is bug free
            n_indifferent_labels = Counter(mu_batch).get(0.5, 0)
            loss_lower_bound = n_indifferent_labels * math.log(2)
            writer2.add_scalar('6.reward_model_loss/label_{}'.format(i_label), loss_lower_bound, epoch)
    # logging.info("reward_model weight after  train {}: {}".format(i_label, list(reward_model.parameters())[0][0][0]))
    return reward_model
    

def init_rm(args):
    """Intitialises and returns the necessary objects for
       reward learning: reward model and optimizer.
    """
    logging.info('Initialising reward model')
    if args.size_rm_ensemble >= 2:
        reward_model = RewardModelEnsemble(args.obs_shape, args.act_shape, args)
    else:
        reward_model = RewardModel(args.obs_shape, args.act_shape, args)
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm)
    return reward_model, optimizer_rm


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
        self.mean_prefs = 0 # mean of reward model across prefs_buffer
        self.var_prefs = 1 # var of reward model across prefs_buffer

    def forward(self, x, normalise=False):
        r_hat = self.layers(x)
        if normalise:
            r_hat = (r_hat - self.mean_prefs) / np.sqrt(self.var_prefs + 1e-8)
        return r_hat


class RewardModelEnsemble(nn.Module):
    """Parameterises r_hat : states x actions -> real rewards
       Approximation of true reward, trained by supervised learning
       on preferences over trajectory segments as in Christiano et al. 2017
       Ouput is an average of `args.size_rm_ensemble` networks
       TODO double check this is how Christiano actually implements it.
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
            setattr(self, 'mean_prefs{}'.format(ensemble_num), 0) # mean of each net in ensemble across prefs_buffer
            setattr(self, 'var_prefs{}'.format(ensemble_num), 1) # var of each net in ensemble across prefs_buffer
        
    def forward(self, x, normalise=False):
        """Returns the average output from forward pass
           through each network in the ensemble.
        """
        output = 0
        for ensemble_num in range(self.ensemble_size):
            net = getattr(self, 'layers{}'.format(ensemble_num))
            r_hat = net(x)
            if normalise:
                mean = getattr(self, 'mean_prefs{}'.format(ensemble_num))
                var = getattr(self, 'var_prefs{}'.format(ensemble_num))
                r_hat = (r_hat - mean) / np.sqrt(var + 1e-8)
            output += r_hat
        return output / self.ensemble_size

    def forward_all(self, x, normalise=False):
        """Instead of averaging output across `ensemble_size`
           networks, return tensor of the output from each network
           in ensemble. Results from different ensembles are
           concatenated on the innermost dimension.
           This will be used, for example, in the BALD algo.
        """
        outputs = []
        for ensemble_num in range(self.ensemble_size):
            net = getattr(self, 'layers{}'.format(ensemble_num))
            r_hat = net(x)
            if normalise:
                mean = getattr(self, 'mean_prefs{}'.format(ensemble_num))
                var = getattr(self, 'var_prefs{}'.format(ensemble_num))
                r_hat = (r_hat - mean) / np.sqrt(var + 1e-8)
            outputs.append(r_hat)
        return torch.cat(outputs, dim=-1)

    def forward_single(self, x, normalise=False):
        """Instead of averaging output across `ensemble_size`
           networks, return output from just one of the forward
           passes, selected u.a.r. from all nets in ensemble.
        """
        ensemble_num = random.randrange(self.ensemble_size)
        net = getattr(self, 'layers{}'.format(ensemble_num))
        r_hat = net(x)
        if normalise:
            mean = getattr(self, 'mean_prefs{}'.format(ensemble_num))
            var = getattr(self, 'var_prefs{}'.format(ensemble_num))
            r_hat = (r_hat - mean) / np.sqrt(var + 1e-8)
        return r_hat


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
    """Clean, assert-free version of the above.
    """
    # import ipdb
    # ipdb.set_trace()
    exp_sum_r_hats_batch = r_hats_batch.sum(dim=2).exp()
    p_hat_12_batch = exp_sum_r_hats_batch[:, 0] / exp_sum_r_hats_batch.sum(dim=1)
    return F.binary_cross_entropy(input=p_hat_12_batch, target=mu_batch, reduction='sum')
        
def compute_loss_rm_ensemble(r_hats_batch_draw, mu_batch):
    """When reward model is an ensemble, you should
       average across elements in ensemble after mapping
       from reward space to preference space, because
       this effectively does the normalisation
       per reward model for free (via the softmax)!
    """
    # import ipdb
    # ipdb.set_trace()
    batch_size, _, clip_length, num_samples = r_hats_batch_draw.shape
    assert r_hats_batch_draw.shape[1] == 2
    exp_sum_r_hats_batch_draw = r_hats_batch_draw.sum(dim=2).exp()
    assert exp_sum_r_hats_batch_draw.shape == (batch_size, 2, num_samples)
    p_hat_12_batch_draw = exp_sum_r_hats_batch_draw[:, 0, :] / exp_sum_r_hats_batch_draw.sum(dim=1)
    assert p_hat_12_batch_draw.shape == (batch_size, num_samples)
    p_hat_12_batch = p_hat_12_batch_draw.mean(1)
    assert p_hat_12_batch.shape == mu_batch.shape
    return F.binary_cross_entropy(input=p_hat_12_batch, target=mu_batch, reduction='sum')


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

    def sample(self, desired_batch_size):
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
        batch_size = min(desired_batch_size, self.current_length)
        idx = np.random.choice(self.current_length, size=batch_size, replace=False)
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
        mean, var = all_rewards_flat.mean(), all_rewards_flat.var()
        if var == 0:
            var = 1
            logging.warning("Variance of true rewards of experience in prefs_buffer is zero!")
            logging.warning("HACK: to save you, I set the variance to 1, but be warned!")
            logging.info("This should be fine as long as RL_baseline is False")
            logging.info("Since in this case, normalised true rewards are logged but not used by the algo")
            logging.info("But if RL_baseline, then you should think carefully about whether hacking var to 1 could lead to strange behaviour")
            # sometimes early in training using RL baseline
            # the prefs buffer won't contain any experience
            # with non-zero true reward. this would make the
            # normalised true rewards explode
            # setting var to 1 if this is the case is one fix
            # though it is kinda hacky & it seems plausible
            # that this could lead to weird behaviour...
            # in general: take care using normalisation with
            # RL baseline! (When you start Atari experiments,
            # best to turn it off initially...)
        return mean, var


def compute_reward_stats(reward_model, prefs_buffer):
    """Returns mean and variance of true and predicted reward
       over the current examples in `prefs_buffer`
       (for normalising rewards sent to agent)
    """
    rt_mean, rt_var = prefs_buffer.compute_mean_var_GT()
    reward_model = compute_mean_var(reward_model, prefs_buffer)
    return (rt_mean, rt_var), reward_model


def compute_mean_var(reward_model, prefs_buffer):
    """Given reward function r and an instance of PrefsBuffer,
       computes E[r(s,a)] and Var[r(s,a)]
       where the expectation and variance are over all the (s,a) pairs
       currently in the buffer (prefs_buffer.clip_pairs).
       Saves them as the appropriate attributes of `reward_model` and
       returns it.
    """
    # flatten the clip_pairs and chuck them through the reward function
    sa_pairs = prefs_buffer.all_flat_sa_pairs()
    reward_model.eval() # turn off dropout
    if isinstance(reward_model, RewardModelEnsemble):
        for ensemble_num in range(reward_model.ensemble_size):
            net = getattr(reward_model, 'layers{}'.format(ensemble_num))
            r_hats = net(sa_pairs).detach().squeeze()
            assert r_hats.shape == (prefs_buffer.current_length * 2 * prefs_buffer.clip_length,)
            mean, var = r_hats.mean().item(), r_hats.var().item()
            if var == 0:
                var = 1
                logging.warning("Variance of predicted rewards over experience in prefs_buffer is zero!")
                logging.warning("HACK: to save you, I set reward_model.var_prefs to 1, but be warned!")
            setattr(reward_model, 'mean_prefs{}'.format(ensemble_num), mean)
            setattr(reward_model, 'var_prefs{}'.format(ensemble_num), var)
    elif isinstance(reward_model, RewardModel):
        r_hats = reward_model(sa_pairs).detach().squeeze()
        assert r_hats.shape == (prefs_buffer.current_length * 2 * prefs_buffer.clip_length,)
        mean, var = r_hats.mean().item(), r_hats.var().item()
        if var == 0:
            var = 1
            logging.warning("Variance of predicted rewards over experience in prefs_buffer is zero!")
            logging.warning("HACK: to save you, I set reward_model.var_prefs to 1, but be warned!")
        reward_model.mean_prefs = mean
        reward_model.var_prefs = var
    return reward_model


def test_correlation(reward_model, env, q_net, args, writer1, i_train_round):
    """TODO Work out what dataset we should eval correlation on... currently
       I use the current q_net to generate rollouts and eval on those.
       But this seems bad b/c the dataset changes every round. And indeed,
       correlation seems to go down as training continues, which seems wrong.
    """
    logging.info('Reward model training complete... Evaluating reward model correlation on {} state-action pairs, accumulated on {} rollouts of length {}'.format(
        args.corr_rollout_steps * args.corr_num_rollouts, args.corr_num_rollouts, args.corr_rollout_steps))
    r_xy, plots = eval_rm_correlation(reward_model, env, q_net, args, args.obs_shape, args.act_shape, rollout_steps=args.corr_rollout_steps, num_rollouts=args.corr_num_rollouts)
    log_correlation(r_xy, plots, writer1, round_num=i_train_round)


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
    # reward_model.train() # we'll now draw samples from posterior to get error bars on scatter plot
    # r_pred_samples = np.array([reward_model(sa_pairs).detach().squeeze().numpy() for _ in range(100)])
    # # dims: 0=samples, 1=examples

    # r_pred_samples_norm = (r_pred_samples - r_pred.mean()) / np.sqrt(r_pred.var() + 1e-8)
    # tenth, ninetieth = np.percentile(r_pred_samples_norm, 10, axis=0), np.percentile(r_pred_samples_norm, 90, axis=0)
    # assert tenth.shape == (num_rollouts * rollout_steps,)
    # assert ninetieth.shape == (num_rollouts * rollout_steps,)
    # minus_err = r_pred_norm - tenth
    # assert (minus_err >= 0).all()
    # plus_err = ninetieth - r_pred_norm

    # scatter_plots['norm_error_bars'] = plt.figure(figsize=(15,8))
    # plt.title('Correlation of (normalised) predicted and true reward with error bars. r_xy = {:.2f}\nAccumulated over {} steps'.format(
    #     r_xy, args.corr_rollout_steps * args.corr_num_rollouts))
    # # maybe do something later about changing colour of reward and error bars corresp. to out-of-distribution state-action pairs
    # plt.errorbar(r_true_norm, r_pred_norm, yerr=np.array([plus_err, minus_err]), fmt='o', ecolor='g', alpha=0.3, capsize=5) # would be nicer to make errors(_norm) a namedtuple (or namedlist if plt.errorbar yerr arg needs a list), s.t. it can be passed straight in, here
    # plt.xlabel('True reward (normalised)')
    # plt.ylabel('Modelled reward (normalised)')

    return r_xy, scatter_plots


def log_correlation(r_xy, plots, writer, round_num):
    writer.add_scalar('10.r_xy', r_xy, round_num)
    writer.add_figure('4.alignment/1-non-norm', plots['non-norm'], round_num) # we can have mutliple figures with the same tag and scroll through them!
    writer.add_figure('4.alignment/2-norm', plots['norm'], round_num)
    # writer.add_figure('4.alignment/3-norm_error_bars', plots['norm_error_bars'], round_num)