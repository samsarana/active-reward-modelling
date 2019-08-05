"""Functions to test a trained policy"""

import numpy as np
import torch, gym, time

def test_policy(q_net, reward_model, reward_stats, args, render=False, num_episodes=100):
    """Using the non-continuous version of the environment and q_net
       with argmax policy (deterministic), run the polcy for
       `num_episodes` and log mean episode return.
       Also log predicted and normalised return
    """
    # set up testing
    env = gym.make(args.env_ID_test)
    env.seed(args.random_seed)
    state, n = env.reset(), 0
    returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
               'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}
    rt_mean, rt_var = reward_stats
    while n < num_episodes:
        if render and n < 3: # if render, watch 3 episodes
            env.render()
            time.sleep(1e-3)
        # agent interact with env
        action = q_net.act(state, epsilon=0)
        assert env.action_space.contains(action)
        next_state, r_true, done, _ = env.step(action)
        # save true reward...
        returns['ep']['true'] += r_true
        returns['ep']['true_norm'] += (r_true - rt_mean) / np.sqrt(rt_var + 1e-8)
        # ...and reward the agent thinks it's getting
        if not args.RL_baseline:
            reward_model.eval() # dropout off at test time
            sa_pair = torch.tensor(np.append(state, action)).float()
            r_pred = reward_model(sa_pair).item()
            r_pred_norm = reward_model(sa_pair, normalise=True).item()
            returns['ep']['pred'] += r_pred
            returns['ep']['pred_norm'] += r_pred_norm
        
        # prepare for next step
        state = next_state
        if done:
            for key, value in returns['ep'].items():
                returns['all'][key].append(value)
                returns['ep'][key] = 0
            state = env.reset()
            n += 1
            if render and n == 3:
                env.close()
    
    assert len(returns['all']['true']) == num_episodes
    return returns['all']

def log_tested_policy(returns, writers, returns_summary, args, i_run, i_train_round):
    """Write test returns to Tensborboard and DataFrame
    """
    writer1, writer2 = writers
    num_test_episodes = len(returns['true'])
    mean_ret_true = np.sum(np.array(returns['true'])) / num_test_episodes
    mean_ret_true_norm = np.sum(np.array(returns['true_norm'])) / num_test_episodes
    returns_summary[i_run][('1.true', i_train_round)] = mean_ret_true # dict format that is friendly to creating a multiindex pd.DataFrame downstream
    returns_summary[i_run][('3.true_norm', i_train_round)] = mean_ret_true_norm
    writer1.add_scalar('1a.mean_ep_return_per_training_round', mean_ret_true, i_train_round)
    writer1.add_scalar('1b.mean_ep_return_per_training_round_normalised', mean_ret_true_norm, i_train_round)
    if not args.RL_baseline:
        mean_ret_pred = np.sum(np.array(returns['pred'])) / num_test_episodes
        mean_ret_pred_norm = np.sum(np.array(returns['pred_norm'])) / num_test_episodes
        returns_summary[i_run][('2.pred', i_train_round)] = mean_ret_pred
        returns_summary[i_run][('4.pred_norm', i_train_round)] = mean_ret_pred_norm
        writer2.add_scalar('1a.mean_ep_return_per_training_round', mean_ret_pred, i_train_round)
        writer2.add_scalar('1b.mean_ep_return_per_training_round_normalised', mean_ret_pred_norm, i_train_round)
    return mean_ret_true

def test_and_log_random_policy(writers, returns_summary, args, i_run, i_train_round, render=False, num_episodes=100):
    """Using the non-continuous version of the environment,
       take random steps for `num_episodes`
       and log mean episode return.
       Also log predicted and normalised return.
       Basically the same as test_policy() + log_tested_policy()
       but since a random agent
       is so simple, putting in all the `if`
       statements s.t. I can use the same function to test them
       seemed like a waste of time.
       However, this code design is nonethless super ugly.
       TODO design a better abstraction for using different agents
       and using/not using GT rewards
    """
    # set up testing
    env = gym.make(args.env_ID_test)
    env.seed(args.random_seed)
    env.reset()
    n = 0
    returns = {'ep': 0, 'all': []}
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)
        # agent interact with env
        action = env.action_space.sample()
        _, r_true, done, _ = env.step(action) # one continuous episode
        returns['ep'] += r_true # record step info
        
        # prepare for next step
        if done:
            returns['all'].append(returns['ep'])
            returns['ep'] = 0
            env.reset()
            n += 1
    
    # log mean return
    assert len(returns['all']) == num_episodes
    writer1, writer2 = writers
    mean_ret_true = np.sum(np.array(returns['all'])) / num_episodes
    returns_summary[i_run][('true', i_train_round)] = mean_ret_true
    writer1.add_scalar('1a.mean_ep_return_per_training_round', mean_ret_true, i_train_round)