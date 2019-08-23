import numpy as np
import matplotlib.pyplot as plt
import torch, logging
import torch.optim as optim
from reward_learning import RewardModel, RewardModelEnsemble, CnnRewardModel

def update_running_mean_var(reward_model, acquired_clip_data):
    reward_model.eval() # turn off dropout
    clip_pairs, _, _ = acquired_clip_data
    _, _, clip_length, obs_act_shape = clip_pairs.shape
    for clip_pair in clip_pairs:
        clip_pair_tensor = torch.from_numpy(clip_pair).unsqueeze(0).float() # unsqueeze first to get batch dimension so it's compatible with mode='clip_pair_batch'
        if isinstance(reward_model, RewardModelEnsemble):
            raise NotImplementedError
        elif isinstance(reward_model, RewardModel) or isinstance(reward_model, CnnRewardModel):
            r_hats = reward_model(clip_pair_tensor, mode='clip_pair_batch').detach().reshape(-1).numpy()
            assert r_hats.shape == (2 * clip_length,)
            for r_hat in r_hats:
                reward_model.running_stats.push(r_hat)
        else:
            raise NotImplementedError("I don't understand reward models of type {}".format(type(reward_model)))
    return reward_model


def compute_mean_var(reward_model, prefs_buffer):
    """Given reward function r and an instance of PrefsBuffer,
       computes E[r(s,a)] and Var[r(s,a)]
       where the expectation and variance are over all the (s,a) pairs
       currently in the buffer (prefs_buffer.clip_pairs).
       Saves them as the appropriate attributes of `reward_model` and
       returns it.
    """
    assert False, "This function is deprecated!"
    # flatten the clip_pairs and chuck them through the reward function
    sa_pairs = prefs_buffer.all_flat_sa_pairs()
    reward_model.eval() # turn off dropout
    if isinstance(reward_model, RewardModelEnsemble):
        for ensemble_num in range(reward_model.ensemble_size):
            net = getattr(reward_model, 'layers{}'.format(ensemble_num))
            r_hats = net(sa_pairs).detach().squeeze() # TODO call net(., mode='batch' instead)
            assert r_hats.shape == (prefs_buffer.current_length * 2 * prefs_buffer.clip_length,)
            mean, var = r_hats.mean().item(), r_hats.var().item()
            if var == 0:
                var = 1
                logging.warning("Variance of predicted rewards over experience in prefs_buffer is zero!")
                logging.warning("HACK: to save you, I set reward_model.var_prefs to 1, but be warned!")
            setattr(reward_model, 'mean_prefs{}'.format(ensemble_num), mean)
            setattr(reward_model, 'var_prefs{}'.format(ensemble_num), var)
    elif isinstance(reward_model, RewardModel) or isinstance(reward_model, CnnRewardModel):
        r_hats = reward_model(sa_pairs, mode='batch').detach().squeeze()
        assert r_hats.shape == (prefs_buffer.current_length * 2 * prefs_buffer.clip_length,)
        mean, var = r_hats.mean().item(), r_hats.var().item()
        if var == 0:
            var = 1
            logging.warning("Variance of predicted rewards over experience in prefs_buffer is zero!")
            logging.warning("HACK: to save you, I set reward_model.var_prefs to 1, but be warned!")
        reward_model.mean_prefs = mean
        reward_model.var_prefs = var
    return reward_model


def save_reward_model(reward_model, rm_optimizer, i_round, args):
    path = '{}/checkpts/rm/{}.pt'.format(args.logdir, i_round)
    torch.save({
        'rm_state_dict': reward_model.state_dict(),
        'rm_optimizer_state_dict': rm_optimizer.state_dict(),
        }, path)


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
    r_pred = reward_model(sa_pairs, mode='batch').detach().squeeze().numpy() # TODO this line will now break with CnnRewardModel. Pass a batch of flat pairs thru
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