import numpy as np
import torch, logging

def log_agent_step(sa_pair, r_true, rets, true_reward_stats, reward_model, args):
    rets['ep']['true'] += r_true
    if args.normalise_rewards:
        assert true_reward_stats != None, "You told me to normalise true reward but haven't told me their mean and var!"
        # rt_mean, rt_var = reward_stats
        rets['ep']['true_norm'] += (r_true - true_reward_stats.mean) / np.sqrt(true_reward_stats.var + 1e-8)
    # also log reward the agent thinks it's getting according to current reward_model
    if not args.RL_baseline:
        reward_model.eval() # dropout off at 'test' time i.e. when logging performance
        sa_tensor = torch.tensor(sa_pair).float()
        r_pred = reward_model(sa_tensor, mode='single').detach().item()
        r_pred_norm = reward_model(sa_tensor, mode='single', normalise=True).detach().item()
        rets['ep']['pred'] += r_pred
        rets['ep']['pred_norm'] += r_pred_norm
    return rets

def log_agent_episode(rets, writers, step, i_train_round, sub_round, args, is_test):
    writer1, writer2 = writers
    if is_test:
        tag = '2a.test_ep_return_per_step/round_{}/test_no_{}'.format(i_train_round, sub_round)
        tag_norm = '2b.test_ep_return_per_step_normalised/round_{}/test_no_{}'.format(i_train_round, sub_round)
    else: # logging training episode
        tag = '4a.train_ep_return_per_step/round_{}'.format(i_train_round)
        tag_norm = '4b.train_ep_return_per_step_normalised/round_{}'.format(i_train_round)
    # interpreting writers: 1 == blue == true!
    writer1.add_scalar(tag, rets['ep']['true'], step)
    if args.normalise_rewards:
        writer1.add_scalar(tag_norm, rets['ep']['true_norm'], step)
    if not args.RL_baseline:
        writer2.add_scalar(tag, rets['ep']['pred'], step)
        writer2.add_scalar(tag_norm, rets['ep']['pred_norm'], step)
    for key, value in rets['ep'].items():
        rets['all'][key].append(value)
        rets['ep'][key] = 0
    return rets

def log_RL_loop(returns, args, i_train_round, sub_round, writers):
    """TODO refactor this funciton with the next one (log_tested_policy)
    """
    i_train_sub_round = args.agent_test_frequency * (i_train_round-1) + sub_round
    writer1, writer2 = writers
    mean_true_returns = np.sum(np.array(returns['all']['true'])) / len(returns['all']['true'])
    mean_true_returns_norm = np.sum(np.array(returns['all']['true_norm'])) / len(returns['all']['true_norm'])
    writer1.add_scalar('3a.train_mean_ep_return_per_sub_round', mean_true_returns, i_train_sub_round)
    if args.normalise_rewards:
        writer1.add_scalar('3b.train_mean_ep_return_per_sub_round_normalised', mean_true_returns_norm, i_train_sub_round)
    if not args.RL_baseline:
        mean_pred_returns = np.sum(np.array(returns['all']['pred'])) / len(returns['all']['pred'])
        mean_pred_returns_norm = np.sum(np.array(returns['all']['pred_norm'])) / len(returns['all']['pred_norm'])
        writer2.add_scalar('3a.train_mean_ep_return_per_sub_round', mean_pred_returns, i_train_sub_round)
        writer2.add_scalar('3b.train_mean_ep_return_per_sub_round_normalised', mean_pred_returns_norm, i_train_sub_round)
    # if sub_round == args.agent_test_frequency - 1: # final sub_round of the round
    #     writer1.add_scalar('3_.train_mean_ep_return_per_round', mean_true_returns, i_train_round)
    #     if not args.RL_baseline:
    #         writer2.add_scalar('3_.train_mean_ep_return_per_round', mean_pred_returns, i_train_round)


def log_tested_policy(returns, writers, returns_summary, args, i_run, i_train_round, sub_round, env):
    """Write test returns to Tensorboard and `returns_summary` DataFrame.
    """
    writer1, writer2 = writers
    i_train_sub_round = args.agent_test_frequency * (i_train_round-1) + sub_round
    num_test_episodes = len(returns['true'])
    mean_ret_true = np.sum(np.array(returns['true'])) / num_test_episodes
    returns_summary[i_run][('1.true', i_train_round, sub_round)] = mean_ret_true # dict format that is friendly to creating a multiindex pd.DataFrame downstream
    writer1.add_scalar('1a.test_mean_ep_return_per_sub_round', mean_ret_true, i_train_sub_round)
    if args.normalise_rewards:
        mean_ret_true_norm = np.sum(np.array(returns['true_norm'])) / num_test_episodes
        returns_summary[i_run][('3.true_norm', i_train_round, sub_round)] = mean_ret_true_norm
        writer1.add_scalar('1b.test_mean_ep_return_per_sub_round_normalised', mean_ret_true_norm, i_train_sub_round)
    if not args.RL_baseline:
        mean_ret_pred = np.sum(np.array(returns['pred'])) / num_test_episodes
        mean_ret_pred_norm = np.sum(np.array(returns['pred_norm'])) / num_test_episodes
        returns_summary[i_run][('2.pred', i_train_round, sub_round)] = mean_ret_pred
        returns_summary[i_run][('4.pred_norm', i_train_round, sub_round)] = mean_ret_pred_norm
        writer2.add_scalar('1a.test_mean_ep_return_per_sub_round', mean_ret_pred, i_train_sub_round)
        writer2.add_scalar('1b.test_mean_ep_return_per_sub_round_normalised', mean_ret_pred_norm, i_train_sub_round)
    # if sub_round == args.agent_test_frequency - 1 or (not args.continue_once_solved \
    #     and env.spec.reward_threshold != None and mean_ret_true >= env.spec.reward_threshold): # final sub_round of the round
    #     writer1.add_scalar('1_.test_mean_ep_return_per_round', mean_ret_true, i_train_round)
    #     if not args.RL_baseline:
    #         writer2.add_scalar('1_.test_mean_ep_return_per_round', mean_ret_pred, i_train_round)
    return mean_ret_true


def save_policy(q_net, policy_optimizer, i_round, i_sub_round, args):
    path = '{}/checkpts/agent/{}-{}.pt'.format(args.logdir, i_round, i_sub_round)
    torch.save({
        'policy_state_dict': q_net.state_dict(),
        'policy_optimizer_state_dict': policy_optimizer.state_dict(),
        }, path)


def log_agent_training_info(args, i_train_round):
    if args.RL_baseline:
        logging.info('Stage {}.1: RL using *true reward*'.format(i_train_round))
    else:
        logging.info('Stage {}.1: RL using reward model'.format(i_train_round))
    if args.normalise_rewards:
        logging.info('Agent will received rewards that have been *normalised* across all s-a pairs currently in prefs_buffer')
    else:
        logging.info('**Agent will receive non-normalised rewards**')
    logging.info('Agent takes {} steps'.format(args.n_agent_steps))
    logging.info('We make a learning update every {} steps(s), from the {}th step onwards'.format(
        args.agent_gdt_step_period, args.agent_learning_starts))
    logging.info('Mean episode return will be tested {} time(s) over the course of this training i.e. every {} agent steps'.format(
        args.agent_test_frequency, args.n_agent_steps // args.agent_test_frequency))