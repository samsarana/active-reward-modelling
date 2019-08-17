"""Functions to test a trained policy"""

import numpy as np
import torch, gym
from gym import wrappers
from time import time, sleep
from rl_logging import *
from atari_preprocessing import *

def test_policy(q_net, reward_model, reward_stats, args, writers, i_train_round, sub_round, num_episodes=100):
    """Using the non-continuous version of the environment and q_net
       with argmax policy (deterministic), run the polcy for
       `num_episodes` and log mean episode return.
       Also log predicted and normalised return
    """
    # set up testing
    env = gym.make(args.env_ID_test)
    # if isinstance(env.env, gym.envs.atari.AtariEnv):
    if args.env_str == 'frozen_lake':
        env = DiscreteToBox(env)
    if args.env_str == 'pong':
        env = preprocess_atari_env(env)
    if args.save_video:
        fname = '{}/videos/test/round={}sub={}time={}/'.format(args.logdir, i_train_round, sub_round, str(time()))
        env = wrappers.Monitor(env, fname) # save all test videos
    env.seed(args.random_seed)
    state, n, step = env.reset(), 0, 0
    returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
               'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}
    while n < num_episodes:
        # agent interact with env
        action = q_net.act(state, epsilon=0)
        assert env.action_space.contains(action)
        next_state, r_true, done, _ = env.step(action)
        # save true reward...
        # sa_pair = torch.tensor(np.append(state, action)).float()
        sa_pair = np.append(state, action).astype(args.oa_dtype, casting='unsafe')
        assert (sa_pair == np.append(state, action)).all() # check casting done safely. should be redundant since i set oa_dtype based on env, earlier. but you can never be too careful since this would fail silently!
        returns = log_agent_step(sa_pair, r_true, returns, reward_stats, reward_model, args)   
        # prepare for next step
        state = next_state
        if done:
            returns = log_agent_episode(returns, writers, step, i_train_round, sub_round, args, is_test=True)
            n += 1
            if args.save_video and args.n_test_vids_to_save > -1 and n == args.n_test_vids_to_save:
                env.close() # stop saving video (unsure whether this unwraps the env or just forces video to stop by closing render window, but it works.)
            state = env.reset()
        step += 1

    assert len(returns['all']['true']) == num_episodes
    return returns['all']


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
    returns_summary[i_run][('true', i_train_round, 0)] = mean_ret_true # 0 b/c sub_round=0 when using random policy
    writer1.add_scalar('1a.test_mean_ep_return_per_round', mean_ret_true, i_train_round)