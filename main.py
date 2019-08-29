import random, argparse, logging, os
import numpy as np
import pandas as pd
from collections import OrderedDict
import gym, gym_barm
import torch
from torch.utils.tensorboard import SummaryWriter

from training_protocol import *
from q_learning import *
from reward_learning import *
from active_learning import *
from atari_preprocessing import *    
from arg_setting import *

def run_experiment(args, i_run, returns_summary):
    # for reproducibility
    args.random_seed = i_run + args.seed_offset
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # TensorBoard logging
    args.logdir = './logs/{}/{}'.format(args.info, args.random_seed)
    writer1 = SummaryWriter(log_dir=args.logdir+'/true')
    writer2 = SummaryWriter(log_dir=args.logdir+'/pred')
    writers = [writer1, writer2]
    # make dirs for saving models at checkpoints
    os.makedirs('{}/checkpts/agent/'.format(args.logdir), exist_ok=True)
    os.makedirs('{}/checkpts/rm/'.format(args.logdir), exist_ok=True)
    os.makedirs('{}/checkpts/prefs/'.format(args.logdir), exist_ok=True)

    # make environment
    env = gym.make(args.env_ID, **args.env_kwargs)
    if args.env_str == 'frozen_lake':
        env = DiscreteToBox(env)
    # if isinstance(env.env, gym.envs.atari.AtariEnv):
    # if args.env_str == 'pong':
    #     env = preprocess_atari_env(env)
    env.seed(args.random_seed)
    if isinstance(env.observation_space, gym.spaces.Box):
        args.obs_shape = env.observation_space.shape[0] # env.observation_space is Box(4,) and calling .shape returns (4,) [gym can be ugly]
        args.obs_shape_all = env.observation_space.shape # TODO ugly
    # elif isinstance(env.observation_space, gym.spaces.Discrete):
    #     args.obs_shape = 1
    else:
        raise RuntimeError("I don't know what observation space {} is!".format(env.observation_space))
    if isinstance(env.action_space, gym.spaces.Discrete):
        args.act_shape = env.action_space.n # [gym doesn't have a nice way to get shape of Discrete space... env.action_space.shape -> () ]
    else:
        raise NotImplementedError('Only discrete actions supported at the moment, for DQN')
    args.obs_act_shape = args.obs_shape + args.act_shape
    args.n_actions = env.action_space.n
    obs = env.reset() # get an obs in order to set args.oa_dtype
    args.oa_dtype = obs.dtype
    # check that arrays for holding ob-act pairs have enough capacity
    if np.issubdtype(args.oa_dtype, np.integer):
        assert max(max(env.observation_space.high), args.n_actions) - 1 <= np.iinfo(args.oa_dtype).max # first max is over the dimensions of observation space. high gives highest value for each dim. second max is over that and number of actions
    elif np.issubdtype(args.oa_dtype, np.floating):
        assert max(max(env.observation_space.high), args.n_actions) - 1 <= np.finfo(args.oa_dtype).max
    else:
        raise RuntimeError("I don't understand the datatype of observations!")

    # setup acquistion function based on args
    active_methods_to_acq_funcs = {
            'BALD': acq_BALD,
            'mean_std': acq_mean_std,
            'max_entropy': acq_max_entropy,
            'var_ratios': acq_var_ratios,
            None: acq_random
        }
    try:
        args.acquistion_func = active_methods_to_acq_funcs[args.active_method]
    except KeyError:
        logging.exception("You specified {} as the active_method type, but I don't know what that is!".format(args.active_method))
        raise

    if args.random_policy:
        do_random_experiment(env, args, returns_summary, writers, i_run)
    else:      
        training_protocol(env, args, writers, returns_summary, i_run)
    
    writer1.close()
    writer2.close()

def main():
    args = parse_arguments()
    os.makedirs('./logs/', exist_ok=True)
    logging.basicConfig(filename='./logs/{}.log'.format(args.info), level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler()) # makes messages print to stderr, too
    # logging.basicConfig(filename='./logs/{}_debug.log'.format(args.info), level=logging.DEBUG)
    logging.info('Running experiment with the following settings:')
    for arg in vars(args):
        logging.info('{}: {}'.format(arg, getattr(args, arg)))
    
    returns_summary = OrderedDict({i: {} for i in range(args.n_runs)})
    for i_run in range(args.n_runs):
        try:
            logging.info('RUN {}/{} BEGIN\n'.format(i_run, args.n_runs - 1))
            run_experiment(args, i_run, returns_summary)
            logging.info('RUN {}/{} SUCCEEDED\n'.format(i_run, args.n_runs - 1))
            pd.DataFrame(returns_summary).to_csv('./logs/{}.csv'.format(args.info), index_label=['ep return type', 'round no.', 'test no.'])
        except SystemExit:
            logging.info('ENVIRONMENT SOLVED!')
            logging.info('RUN {}/{} SUCCEEDED\n'.format(i_run, args.n_runs - 1))
            pd.DataFrame(returns_summary).to_csv('./logs/{}.csv'.format(args.info), index_label=['ep return type', 'round no.', 'test no.'])
        except:
            logging.exception('RUN {}/{} FAILED with the following traceback:\n'.format(i_run, args.n_runs - 1))

if __name__ == '__main__':
    main()