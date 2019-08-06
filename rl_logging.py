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

def log_agent_episode(rets, writers, step, i_train_round, args):
    writer1, writer2 = writers
    # interpreting writers: 1 == blue == true!
    writer1.add_scalar('3a.ep_return_against_step/round_{}'.format(i_train_round), rets['ep']['true'], step+1)
    writer1.add_scalar('3b.ep_return_against_step_normalised/round_{}'.format(i_train_round), rets['ep']['true_norm'], step+1)
    if not args.RL_baseline:
        writer2.add_scalar('3a.ep_return_against_step/round_{}'.format(i_train_round), rets['ep']['pred'], step+1)
        writer2.add_scalar('3b.ep_return_against_step_normalised/round_{}'.format(i_train_round), rets['ep']['pred_norm'], step+1)
    for key, value in rets['ep'].items():
        rets['all'][key].append(value)
        rets['ep'][key] = 0
    return rets