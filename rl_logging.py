import numpy as np

def log_agent_step(sa_pair, r_true, rets, reward_stats, reward_model, args):
    rt_mean, rt_var = reward_stats
    rets['ep']['true'] += r_true
    rets['ep']['true_norm'] += (r_true - rt_mean) / np.sqrt(rt_var + 1e-8)
    # also log reward the agent thinks it's getting according to current reward_model
    if not args.RL_baseline:
        reward_model.eval() # dropout off at 'test' time i.e. when logging performance
        r_pred = reward_model(sa_pair).item()
        r_pred_norm = reward_model(sa_pair, normalise=True).item()
        rets['ep']['pred'] += r_pred
        rets['ep']['pred_norm'] += r_pred_norm
    return rets

def log_agent_episode(rets, writers, step, i_train_round, args, is_test):
    writer1, writer2 = writers
    if is_test:
        tag = '2a.test_ep_return_per_step/round_{}'.format(i_train_round)
        tag_norm = '2b.test_ep_return_per_step_normalised/round_{}'.format(i_train_round)
    else: # logging training episode
        tag = '4a.train_ep_return_per_step/round_{}'.format(i_train_round)
        tag_norm = '4b.train_ep_return_per_step_normalised/round_{}'.format(i_train_round)
    # interpreting writers: 1 == blue == true!
    writer1.add_scalar(tag, rets['ep']['true'], step)
    writer1.add_scalar(tag_norm, rets['ep']['true_norm'], step)
    if not args.RL_baseline:
        writer2.add_scalar(tag, rets['ep']['pred'], step)
        writer2.add_scalar(tag_norm, rets['ep']['pred_norm'], step)
    for key, value in rets['ep'].items():
        rets['all'][key].append(value)
        rets['ep'][key] = 0
    return rets