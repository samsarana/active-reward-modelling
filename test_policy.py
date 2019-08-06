"""Functions to test a trained policy"""

import numpy as np
import torch, gym
from gym import wrappers
from time import time, sleep
from rl_logging import log_agent_step

def test_policy(q_net, reward_model, reward_stats, args, num_episodes=100):
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
        if args.render_policy_test and n < 3: # if render, watch 3 episodes
            env.render()
            sleep(1e-3)
        # agent interact with env
        action = q_net.act(state, epsilon=0)
        assert env.action_space.contains(action)
        next_state, r_true, done, _ = env.step(action)
        # save true reward...
        sa_pair = torch.tensor(np.append(state, action)).float()
        returns = log_agent_step(sa_pair, r_true, returns, reward_stats, reward_model, args)        
        # prepare for next step
        state = next_state
        if done:
            for key, value in returns['ep'].items():
                returns['all'][key].append(value)
                returns['ep'][key] = 0
            n += 1
            if args.render_policy_test and n == 3:
                env.close()
            if n == num_episodes -1 and args.save_video:
                # save the final test episode (see https://github.com/openai/gym/wiki/FAQ#how-do-i-export-the-run-to-a-video-file)
                env = wrappers.Monitor(env, './logs/videos/test/' + str(time()) + '/')
            state = env.reset()
    
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

def test_and_log_random_policy(writers, returns_summary, args, i_run, i_train_round, num_episodes=100):
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
        if args.render_policy_test:
            env.render()
            sleep(1e-3)
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