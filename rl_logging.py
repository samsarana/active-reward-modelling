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


def log_agent_episode(rets, writers, step, i_train_round, args, test_num=None):
    if test_num is None or test_num % 20 == 0: # log all train episodes and 5 test episodes (we fix num test episodes at 100)
        writer1, writer2 = writers
        if test_num is not None: # we're logging a test episode
            tag = '2a.test_ep_return_per_step/round_{}/test_no_{}'.format(i_train_round, test_num)
            tag_norm = '2b.test_ep_return_per_step_normalised/round_{}/test_no_{}'.format(i_train_round, test_num)
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


def log_agent_test(train_returns, test_returns, returns_summary, step, i_test, i_train_round, i_run, writers, args):
    writer1, writer2 = writers
    # log train returns
    mean_true_returns_train = np.sum(np.array(train_returns['all']['true'])) / len(train_returns['all']['true'])
    writer1.add_scalar('3a.train_mean_ep_return_per_step/round_{}'.format(i_train_round), mean_true_returns_train, step)
    if args.normalise_rewards:
        mean_true_returns_train_norm = np.sum(np.array(train_returns['all']['true_norm'])) / len(train_returns['all']['true_norm'])
        writer1.add_scalar('3b.train_mean_ep_return_per_step_normalised/round_{}'.format(i_train_round), mean_true_returns_train_norm, step)
    if not args.RL_baseline:
        mean_pred_returns_train = np.sum(np.array(train_returns['all']['pred'])) / len(train_returns['all']['pred'])
        mean_pred_returns_train_norm = np.sum(np.array(train_returns['all']['pred_norm'])) / len(train_returns['all']['pred_norm'])
        writer2.add_scalar('3a.train_mean_ep_return_per_step/round_{}'.format(i_train_round), mean_pred_returns_train, step)
        writer2.add_scalar('3b.train_mean_ep_return_per_step_normalised/round_{}'.format(i_train_round), mean_pred_returns_train_norm, step)
    # log test returns
    num_test_episodes = len(test_returns['true'])
    mean_ret_true_test = np.sum(np.array(test_returns['true'])) / num_test_episodes
    returns_summary[i_run][('1.true', i_train_round, i_test)] = mean_ret_true_test # dict format that is friendly to creating a multiindex pd.DataFrame downstream
    writer1.add_scalar('1a.test_mean_ep_return_per_step/round_{}'.format(i_train_round), mean_ret_true_test, step)
    if args.normalise_rewards:
        mean_ret_true_test_norm = np.sum(np.array(test_returns['true_norm'])) / num_test_episodes
        returns_summary[i_run][('3.true_norm', i_train_round, i_test)] = mean_ret_true_test_norm
        writer1.add_scalar('1b.test_mean_ep_return_per_step_normalised/round_{}'.format(i_train_round), mean_ret_true_test_norm, step)
    if not args.RL_baseline:
        mean_ret_pred_test = np.sum(np.array(test_returns['pred'])) / num_test_episodes
        mean_ret_pred_test_norm = np.sum(np.array(test_returns['pred_norm'])) / num_test_episodes
        returns_summary[i_run][('2.pred', i_train_round, i_test)] = mean_ret_pred_test
        returns_summary[i_run][('4.pred_norm', i_train_round, i_test)] = mean_ret_pred_test_norm
        writer2.add_scalar('1a.test_mean_ep_return_per_step/round_{}'.format(i_train_round), mean_ret_pred_test, step)
        writer2.add_scalar('1b.test_mean_ep_return_per_step_normalised/round_{}'.format(i_train_round), mean_ret_pred_test_norm, step)
    return mean_ret_true_test


# def log_agent_train_loop(train_returns, test_returns, n_episodes, i_train_round, writers, args):
#     writer1, writer2 = writers
#     # log test returns
#     max_true_rets_train = np.max(np.array(train_returns['all']['true']))
#     writer1.add_scalar('3_.train_max_ep_return_per_round', max_true_rets_train, i_train_round)
#     if args.normalise_rewards:
#         max_true_returns_train_norm = np.max(np.array(train_returns['all']['true_norm']))
#         writer1.add_scalar('3_.train_max_ep_return_per_round_normalised', max_true_returns_train_norm, i_train_round)
#     if not args.RL_baseline:
#         max_pred_returns_train = np.max(np.array(train_returns['all']['pred']))
#         max_pred_returns_train_norm = np.max(np.array(train_returns['all']['pred_norm']))
#         writer2.add_scalar('3_.train_max_ep_return_per_round', max_pred_returns_train, i_train_round)
#         writer2.add_scalar('3_.train_max_ep_return_per_round_normalised', max_pred_returns_train_norm, i_train_round)
#     # log train returns
#     max_ret_true_test = np.max(np.array(test_returns['true']))
#     writer1.add_scalar('1_.test_max_ep_return_per_round', max_ret_true_test, i_train_round)
#     if args.normalise_rewards:
#         mean_ret_true_test_norm = np.max(np.array(test_returns['true_norm']))
#         writer1.add_scalar('1_.test_max_ep_return_per_round_normalised', mean_ret_true_test_norm, i_train_round)
#     if not args.RL_baseline:
#         max_ret_pred_test = np.max(np.array(test_returns['pred']))
#         max_ret_pred_test_norm = np.max(np.array(test_returns['pred_norm']))
#         writer2.add_scalar('1_.test_max_ep_return_per_round', max_ret_pred_test, i_train_round)
#         writer2.add_scalar('1_.test_max_ep_return_per_round_normalised', max_ret_pred_test_norm, i_train_round)
#     writer1.add_scalar('10.n_episodes', n_episodes, i_train_round)


# def log_RL_loop(returns, n_episodes, args, step, i_train_round, writers):
#     """TODO refactor this funciton with the next one (log_tested_policy)
#     """
#     writer1, writer2 = writers
#     mean_true_returns = np.sum(np.array(returns['all']['true'])) / len(returns['all']['true'])
#     mean_true_returns_norm = np.sum(np.array(returns['all']['true_norm'])) / len(returns['all']['true_norm'])
#     writer1.add_scalar('3a.train_mean_ep_return_per_step', mean_true_returns, step)
#     if args.normalise_rewards:
#         writer1.add_scalar('3b.train_mean_ep_return_per_step_normalised', mean_true_returns_norm, step)
#     if not args.RL_baseline:
#         mean_pred_returns = np.sum(np.array(returns['all']['pred'])) / len(returns['all']['pred'])
#         mean_pred_returns_norm = np.sum(np.array(returns['all']['pred_norm'])) / len(returns['all']['pred_norm'])
#         writer2.add_scalar('3a.train_mean_ep_return_per_step', mean_pred_returns, step)
#         writer2.add_scalar('3b.train_mean_ep_return_per_step_normalised', mean_pred_returns_norm, step)
#     # if sub_round == args.agent_test_frequency - 1: # final sub_round of the round
#     #     writer1.add_scalar('3_.train_mean_ep_return_per_round', mean_true_returns, i_train_round)
#     #     if not args.RL_baseline:
#     #         writer2.add_scalar('3_.train_mean_ep_return_per_round', mean_pred_returns, i_train_round)


# def log_tested_policy(returns, writers, returns_summary, args, i_run, i_train_round, i_test, env):
#     """Write test returns to Tensorboard and `returns_summary` DataFrame.
#     """
#     writer1, writer2 = writers
#     num_test_episodes = len(returns['true'])
#     mean_ret_true = np.sum(np.array(returns['true'])) / num_test_episodes
#     returns_summary[i_run][('1.true', i_train_round, i_test)] = mean_ret_true # dict format that is friendly to creating a multiindex pd.DataFrame downstream
#     writer1.add_scalar('1a.test_mean_ep_return_per_step', mean_ret_true, step)
#     if args.normalise_rewards:
#         mean_ret_true_norm = np.sum(np.array(returns['true_norm'])) / num_test_episodes
#         returns_summary[i_run][('3.true_norm', i_train_round, i_test)] = mean_ret_true_norm
#         writer1.add_scalar('1b.test_mean_ep_return_per_step_normalised', mean_ret_true_norm, step)
#     if not args.RL_baseline:
#         mean_ret_pred = np.sum(np.array(returns['pred'])) / num_test_episodes
#         mean_ret_pred_norm = np.sum(np.array(returns['pred_norm'])) / num_test_episodes
#         returns_summary[i_run][('2.pred', i_train_round, i_test)] = mean_ret_pred
#         returns_summary[i_run][('4.pred_norm', i_train_round, i_test)] = mean_ret_pred_norm
#         writer2.add_scalar('1a.test_mean_ep_return_per_step', mean_ret_pred, step)
#         writer2.add_scalar('1b.test_mean_ep_return_per_step_normalised', mean_ret_pred_norm, step)
#     # if sub_round == args.agent_test_frequency - 1 or (not args.continue_once_solved \
#     #     and env.spec.reward_threshold != None and mean_ret_true >= env.spec.reward_threshold): # final sub_round of the round
#     #     writer1.add_scalar('1_.test_mean_ep_return_per_round', mean_ret_true, i_train_round)
#     #     if not args.RL_baseline:
#     #         writer2.add_scalar('1_.test_mean_ep_return_per_round', mean_ret_pred, i_train_round)
#     return mean_ret_true


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