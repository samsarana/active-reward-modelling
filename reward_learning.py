"""Classes and functions to do reward learning"""

import math, random, logging
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import RunningStat

def train_reward_model(reward_model, prefs_buffer, optimizer_rm, args, writers, i_label):
    writer1, writer2 = writers
    epochs = args.n_epochs_pretrain_rm if i_label <= -1 else args.n_epochs_train_rm
    logging.info("Training reward model for {} epochs".format(epochs))
    reward_model.train() # dropout on
    # logging.info("reward_model weight before train {}: {}".format(i_label, list(reward_model.parameters())[0][0][0]))
    for epoch in range(epochs):
        with torch.autograd.detect_anomaly():
            if args.train_rm_ensemble_independently:
                assert isinstance(reward_model, RewardModelEnsemble)
                # independent_train_rm_ensemble()
                loss_total, n_indifferent_labels = 0, 0
                for ensemble_num in range(reward_model.ensemble_size):
                    clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
                    r_hats_batch = reward_model.forward_single(clip_pair_batch, ensemble_num, mode='clip_pair_batch', normalise=False).squeeze(-1)
                    loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
                    optimizer_rm.zero_grad()
                    loss_rm.backward()
                    optimizer_rm.step()
                    # acumulate loss for logging purposes
                    loss_total += loss_rm.detach().item()
                    n_indifferent_labels += Counter(mu_batch.numpy()).get(0.5, 0)
                writer1.add_scalar('6.reward_model_loss/label_{}'.format(i_label), loss_total / reward_model.ensemble_size, epoch)
                # compute lower bound for loss_rm and plot this too
                loss_lower_bound = n_indifferent_labels / reward_model.ensemble_size * math.log(2)
                writer2.add_scalar('6.reward_model_loss/label_{}'.format(i_label), loss_lower_bound, epoch)
            else:
                # get a single minibatch
                clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
                if isinstance(reward_model, RewardModelEnsemble):
                    r_hats_batch_draw = reward_model.forward_all(clip_pair_batch, mode='clip_pair_batch', normalise=False).squeeze(-1)
                    loss_rm = compute_loss_rm_ensemble(r_hats_batch_draw, mu_batch)
                else:
                    assert isinstance(reward_model, RewardModel) or isinstance(reward_model, CnnRewardModel)
                    r_hats_batch = reward_model(clip_pair_batch, mode='clip_pair_batch', normalise=False).squeeze(-1)
                    loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
                optimizer_rm.zero_grad()
                loss_rm.backward()
                optimizer_rm.step()
                writer1.add_scalar('6.reward_model_loss/label_{}'.format(i_label), loss_rm, epoch)
                # compute lower bound for loss_rm and plot this too
                n_indifferent_labels = Counter(mu_batch.numpy()).get(0.5, 0)
                loss_lower_bound = n_indifferent_labels * math.log(2)
                writer2.add_scalar('6.reward_model_loss/label_{}'.format(i_label), loss_lower_bound, epoch)
    # logging.info("reward_model weight after  train {}: {}".format(i_label, list(reward_model.parameters())[0][0][0]))
    return reward_model


class CnnRewardModel(nn.Module):
    """Parameterises r_hat : states x actions -> real rewards
       Use a convolutional NN for the states, then append action
       before pass through fully connected layer (as done in MIRI
       Atari extension of Tom Brown's implemenation).
    """
    def __init__(self, state_size, action_size, args):
        super().__init__()
        if args.rm_archi == 'cnn':
            self.convolutions = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(), # NB TODO Christiano uses leaky ReLU
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 512, kernel_size=7, stride=1),
                nn.ReLU(),
            )
        else:
            assert args.rm_archi == 'cnn_mod'
            self.convolutions = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(), # NB TODO Christiano uses leaky ReLU
                nn.BatchNorm2d(32), # TODO some debate as to order of these layers https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
                nn.Dropout(args.p_dropout_rm),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Dropout(args.p_dropout_rm),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Dropout(args.p_dropout_rm),
                nn.Conv2d(64, 512, kernel_size=7, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Dropout(args.p_dropout_rm),
            )
        self.fc = nn.Sequential(
            nn.Linear(512 + action_size, 512), # NB TODO Christiano uses 64 unit hidden layer
            nn.ReLU(),
            nn.Linear(512, 1)
        ) 
        # self.mean_prefs = 0 # mean of reward model across prefs_buffer
        # self.var_prefs = 1 # var of reward model across prefs_buffer
        self.running_stats = RunningStat()
        self.state_size = state_size
        self.action_size = action_size
        self.clip_length = args.clip_length

    def forward(self, sa_pair, mode, normalise=False):
        """Forward pass for a batch of state-action pairs
           input shape = (B, oa_shape)
           TODO cleaner code design would be to have reward
           model take state, action rather than state-action
           pair. THis would avoid the ugly if-else
           statemenet at the start of this function.
           But I'm running out of time, and this would
           be a relatively major change which would
           force me to test all the code again.
           The simpler version is to still store experience
           in pairs then split it up when we call reward_model()
           according to whether we're passing it single, batch 
           or clip pair batch. But this isn't worth the time
           right now, either.
        """
        if mode == 'single':
            state, action = sa_pair[:self.state_size], sa_pair[-self.action_size:]
        elif mode == 'batch':
            state, action = sa_pair[:, :self.state_size], sa_pair[:, -self.action_size:]
            batch_size = sa_pair.shape[0] # used only in assert
        elif mode == 'clip_pair_batch':
            state, action = sa_pair[:,:,:,:self.state_size], sa_pair[:,:,:,-self.action_size:]
            batch_size = sa_pair.shape[0] # used only in assert
        else:
            raise RuntimeError("I don't know what mode {} is!".format(mode))

        x = state.view(-1, 3, 84, 84) # conv2d needs 4d input so even in the clip_pair_batch case we have to take a view, then reconstruct batch, pair, clip length dims
        x = self.convolutions(x)
        if mode == 'single':
            x = x.view(512)
        elif mode == 'batch':
            x = x.view(-1, 512)
        elif mode == 'clip_pair_batch':
            x = x.view(-1, 2, self.clip_length, 512) # get batch size, pair and clip length dims back
            # TODO I'm far from certain that these reshapes preserve the dimensions. use debugger to check. but how?
            assert action.shape == (batch_size, 2, self.clip_length, self.action_size)
        x = torch.cat((x, action), dim=-1)
        assert x.shape[-1] == (512 + self.action_size)
        r_hat = self.fc(x)
        if normalise:
            r_hat = (r_hat - self.running_stats.mean) / np.sqrt(self.running_stats.var + 1e-8)
        return r_hat   

    # def forward(self, sa_pair, normalise=False):
    #     """Forward pass for a batch of state-action pairs
    #        input shape = (B, oa_shape)
    #     """
    #     # r_hat = self.layers(x)
    #     batch_size = sa_pair.shape[0] # used only in assert
    #     state, action = sa_pair[:, :self.state_size], sa_pair[:, -self.action_size:]
    #     x = state.view(-1, 3, 84, 84)
    #     x = self.convolutions(x)
    #     x = x.view(-1, 512)
    #     x = torch.cat((x, action), dim=-1)
    #     assert x.shape == (batch_size, 512 + self.action_size)
    #     r_hat = self.fc(x)
    #     if normalise:
    #         r_hat = (r_hat - self.mean_prefs) / np.sqrt(self.var_prefs + 1e-8)
    #     return r_hat

    # def forward_sa_pair(self, sa_pair, normalise=False):
    #     """Forward pass for a single state-action pair
    #        input shape = (oa_shape)
    #     """
    #     # r_hat = self.layers(x)
    #     state, action = sa_pair[:self.state_size], sa_pair[-self.action_size:]
    #     x = state.view(-1, 3, 84, 84)
    #     x = self.convolutions(x)
    #     x = x.view(512)
    #     x = torch.cat((x, action), dim=-1)
    #     assert x.shape == (512 + self.action_size,)
    #     r_hat = self.fc(x)
    #     if normalise:
    #         r_hat = (r_hat - self.mean_prefs) / np.sqrt(self.var_prefs + 1e-8)
    #     return r_hat

    # def forward_batch(self, clip_pair_batch, normalise=False):
    #     """Forward pass for a batch of clip pairs
    #        input shape = (B, 2, clip_shape, oa_shape)
    #     """
    #     # r_hat = self.layers(x)
    #     batch_size = clip_pair_batch.shape[0] # used only in asserts
    #     state, action = clip_pair_batch[:,:,:,:self.state_size], clip_pair_batch[:,:,:,-self.action_size:]
    #     # x = state.view(-1, 3, 84, 84)
    #     # x = state.view(-1, 2, self.clip_length, 3, 84, 84) # 2 b/c clip *pair*
    #     x = state.view(-1, 3, 84, 84) # conv2d needs 4d input so we have to take a view then reconstruct batch, pair, clip length dims
    #     x = self.convolutions(x)
    #     x = x.view(-1, 2, self.clip_length, 512) # get batch size, pair and clip length dims back
    #     # TODO I'm far from 100% sure that this stuff preserves the dimensions
    #     assert action.shape == (batch_size, 2, self.clip_length, self.action_size)
    #     x = torch.cat((x, action), dim=-1)
    #     assert x.shape == (batch_size, 2, self.clip_length, 512 + self.action_size)
    #     r_hat = self.fc(x)
    #     if normalise:
    #         r_hat = (r_hat - self.mean_prefs) / np.sqrt(self.var_prefs + 1e-8)
    #     return r_hat


class RewardModel(nn.Module):
    """Parameterises r_hat : states x actions -> real rewards
       Approximation of true reward, trained by supervised learning
       on preferences over trajectory segments as in Christiano et al. 2017
    """
    def __init__(self, state_size, action_size, args):
        """Feedforward NN with 2 hidden layers"""
        super().__init__()
        if args.h3_rm: # 3 hidden layer reward model
            self.layers = nn.Sequential(
                nn.Linear(state_size + action_size, args.h1_rm),
                nn.ReLU(),
                nn.BatchNorm1d(args.h1_rm),
                nn.Dropout(args.p_dropout_rm),
                nn.Linear(args.h1_rm, args.h2_rm),
                nn.ReLU(),
                nn.BatchNorm1d(args.h2_rm),
                nn.Dropout(args.p_dropout_rm),
                nn.Linear(args.h2_rm, args.h3_rm),
                nn.ReLU(),
                nn.BatchNorm1d(args.h3_rm),
                nn.Dropout(args.p_dropout_rm),
                nn.Linear(args.h3_rm, 1)
            )
        else: # 2 hidden layer reward model
            self.layers = nn.Sequential(
                nn.Linear(state_size + action_size, args.h1_rm),
                nn.ReLU(),
                nn.Dropout(args.p_dropout_rm),
                nn.Linear(args.h1_rm, args.h2_rm),
                nn.ReLU(),
                nn.Dropout(args.p_dropout_rm),
                nn.Linear(args.h2_rm, 1)
            )
        self.running_stats = RunningStat()
        self.sa_size = state_size + action_size

    def forward(self, x, mode=None, normalise=False):
        """
        `mode` is unused... this is silly code but see docstr
        of CnnRewardModel.forward() for why I'm doing it this
        way
        """
        x = x.view(-1, self.sa_size)
        x = self.layers(x)
        if normalise:
            x = (x - self.running_stats.mean) / np.sqrt(self.running_stats.var + 1e-8)
        if mode == 'single' or mode == 'batch':
            x = x.view(-1)
        elif mode == 'clip_pair_batch':
            x = x.view(-1, 2, 25) # 25 is BAD code
        return x


class RewardModelEnsemble(nn.Module):
    def __init__(self, state_size, action_size, args):
        super().__init__()
        self.ensemble_size = args.size_rm_ensemble
        assert self.ensemble_size >= 2
        for ensemble_num in range(self.ensemble_size):
            setattr(self, 'layers{}'.format(ensemble_num), 
                    RewardModel(state_size, action_size, args)
            )
            setattr(self, 'running_stats{}'.format(ensemble_num), RunningStat()) # running mean and variance of each net in ensemble across prefs_buffer

    def forward(self, x, mode=None, normalise=False):
        """Returns the average output from forward pass
           through each network in the ensemble.
        """
        output = 0
        for ensemble_num in range(self.ensemble_size):
            net = getattr(self, 'layers{}'.format(ensemble_num))
            r_hat = net(x)
            if normalise:
                running_stats = getattr(self, 'running_stats{}'.format(ensemble_num))
                r_hat = (r_hat - running_stats.mean) / np.sqrt(running_stats.var + 1e-8)
            output += r_hat
        return output / self.ensemble_size

    def forward_all(self, x, mode=None, normalise=False):
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
                running_stats = getattr(self, 'running_stats{}'.format(ensemble_num))
                r_hat = (r_hat - running_stats.mean) / np.sqrt(running_stats.var + 1e-8)
            outputs.append(r_hat)
        return torch.cat(outputs, dim=-1)

    def forward_single(self, x, ensemble_num='random', mode=None, normalise=False):
        """Instead of averaging output across `ensemble_size`
           networks, return output from just one of the forward
           passes, selected u.a.r. from all nets in ensemble.
        """
        if ensemble_num == 'random':
            ensemble_num = random.randrange(self.ensemble_size)
        assert 0 <= ensemble_num <= self.ensemble_size -1
        net = getattr(self, 'layers{}'.format(ensemble_num))
        r_hat = net(x)
        if normalise:
            running_stats = getattr(self, 'running_stats{}'.format(ensemble_num))
            r_hat = (r_hat - running_stats.mean) / np.sqrt(running_stats.var + 1e-8)
        return r_hat


class RewardModelEnsembleOld(nn.Module):
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
        
    def forward(self, x, mode=None, normalise=False):
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

    def forward_all(self, x, mode=None, normalise=False):
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

    def forward_single(self, x, mode=None, normalise=False):
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
    batch_size, _, clip_length, num_samples = r_hats_batch_draw.shape
    assert r_hats_batch_draw.shape[1] == 2
    exp_sum_r_hats_batch_draw = r_hats_batch_draw.sum(dim=2).exp()
    assert exp_sum_r_hats_batch_draw.shape == (batch_size, 2, num_samples)
    p_hat_12_batch_draw = exp_sum_r_hats_batch_draw[:, 0, :] / exp_sum_r_hats_batch_draw.sum(dim=1)
    assert p_hat_12_batch_draw.shape == (batch_size, num_samples)
    p_hat_12_batch = p_hat_12_batch_draw.mean(1)
    assert p_hat_12_batch.shape == mu_batch.shape
    return F.binary_cross_entropy(input=p_hat_12_batch, target=mu_batch, reduction='sum')


def init_rm(args):
    """Intitialises and returns the necessary objects for
       reward learning: reward model and optimizer.
    """
    logging.info('Initialising reward model')
    if args.rm_archi == 'mlp':
        if args.size_rm_ensemble >= 2:
            reward_model = RewardModelEnsemble(args.obs_shape, args.act_shape, args)
        else:
            reward_model = RewardModel(args.obs_shape, args.act_shape, args)
    else:
        assert args.rm_archi == 'cnn' or args.rm_archi == 'cnn_mod'
        if args.size_rm_ensemble >= 2:
            raise NotImplementedError("U haven't yet implemented ensemble of CNN reward models!")
        else:
            reward_model = CnnRewardModel(args.obs_shape, args.act_shape, args)
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm)
    return reward_model, optimizer_rm



class PrefsBuffer():
    def __init__(self, args):
        self.capacity = args.prefs_buffer_size
        self.clip_length = args.clip_length
        self.obs_act_length = args.obs_act_shape
        self.oa_dtype = args.oa_dtype
        self.clip_pairs = np.zeros(shape=(self.capacity, 2, self.clip_length, self.obs_act_length), dtype=args.oa_dtype) # 2 because preference is on clip *pair*
        self.rewards = np.zeros(shape=(self.capacity, 2, self.clip_length))
        self.mus = np.zeros(shape=self.capacity)
        self.current_length = 0 # maintain the current length to help with sampling from the fixed size array

    def push(self, new_clip_pairs, new_rews, new_mus):
        """Takes
            new_clip_pairs.shape == (_, 2, clip_length, obs_act_length)
            new_mus.shape        == (_,)
            and pushes them onto the circular buffers self.clip_pairs
            and self.mus
        """
        len_new_pairs = len(new_clip_pairs)
        i_start = self.current_length
        i_stop = self.current_length + len_new_pairs
        assert i_stop <= self.capacity, "You're trying to add more clips than prefs_buffer has space for!"
        assert len_new_pairs == len(new_mus)
        assert new_clip_pairs.dtype == self.oa_dtype, "Trying to add clip pair with dtype {} but PrefsBuffer only takes dtype {}".format(new_clip_pairs.dtype, self.oa_dtype)
        # self.clip_pairs = np.roll(self.clip_pairs, len_new_pairs, axis=0)
        # self.rewards = np.roll(self.rewards, len_new_pairs, axis=0)
        # self.mus = np.roll(self.mus, len_new_pairs)
        # assert (self.clip_pairs[:len_new_pairs]).all() == 0, "You are about to throw away labels from prefs_buffer!"
        # self.clip_pairs[:len_new_pairs] = new_clip_pairs
        # self.rewards[:len_new_pairs] = new_rews
        # self.mus[:len_new_pairs] = new_mus

        self.clip_pairs[i_start:i_stop] = new_clip_pairs
        self.rewards[i_start:i_stop] = new_rews
        self.mus[i_start:i_stop] = new_mus

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


# def compute_reward_stats(reward_model, prefs_buffer):
#     """Returns mean and variance of true and predicted reward
#        over the current examples in `prefs_buffer`
#        (for normalising rewards sent to agent)
#     """
#     rt_mean, rt_var = prefs_buffer.compute_mean_var_GT()
#     reward_model = compute_mean_var(reward_model, prefs_buffer)
#     return (rt_mean, rt_var), reward_model