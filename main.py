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
from defaults import *
from atari_preprocessing import preprocess_atari_env

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--info', type=str, default='', help='Tensorboard log is saved in ./logs/i_run/random_seed/[true|pred]/')
    parser.add_argument('--default_settings', type=str, default=None, help='Flag to override args with those in default.py. Choice of: acrobot_sam, openai, openai_atari, cartpole')
    parser.add_argument('--env_str', type=str, default='cartpole', help='Choice of: acrobot, mountain_car, cartpole, cartpole_old, cartpole_old_rich')
    parser.add_argument('--n_runs', type=int, default=40, help='number of runs to repeat the experiment')
    parser.add_argument('--n_rounds', type=int, default=40, help='number of rounds to repeat main training loop')
    parser.add_argument('--RL_baseline', action='store_true', help='Do RL baseline instead of reward learning?')
    parser.add_argument('--random_policy', action='store_true', help='Do the experiments with an entirely random policy, to benchmark performance')
    parser.add_argument('--test', action='store_true', help='Flag to make training procedure very short (to check for errors)')
    parser.add_argument('--render_policy_test', action='store_true', help='Flag to render 3 episodes of policy test')
    parser.add_argument('--save_video', action='store_true', help='Flag to save final 2 test episode and final ~3 train episodes')
    parser.add_argument('--continue_once_solved', action='store_true', help='Experiment will continue even when agent test mean ep return >= env.spec.reward_threshold')
    parser.add_argument('--seed_offset', type=int, default=0, help='We seed with i_run + seed_offset, where i_run in {0..n_runs-1}')

    # agent hyperparams
    parser.add_argument('--h1_agent', type=int, default=32)
    parser.add_argument('--h2_agent', type=int, default=64)
    parser.add_argument('--batch_size_agent', type=int, default=32)
    parser.add_argument('--lr_agent', type=float, default=1e-3)
    parser.add_argument('--lambda_agent', type=float, default=1e-4, help='coefficient for L2 regularization for agent optimization')
    parser.add_argument('--replay_buffer_size', type=int, default=30000)
    parser.add_argument('--target_update_period', type=int, default=1) # Ibarz: 8000, but hard updates
    parser.add_argument('--target_update_tau', type=float, default=8e-2) # Ibarz: 1 (hard update)
    parser.add_argument('--agent_gdt_step_period', type=int, default=1) # Ibarz: 4
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='exploration probability for agent at start')
    # parser.add_argument('--epsilon_decay', type=float, default=0.999, help='`epsilon *= epsilon * epsilon_decay` every learning step, until `epsilon_stop`') 
    parser.add_argument('--epsilon_stop', type=float, default=0.01)
    parser.add_argument('--exploration_fraction', type=float, default=0.1, help='Over what fraction of entire training period is epsilon annealed (linearly)?')
    parser.add_argument('--n_labels_per_round', type=int, default=5, help='How many labels to acquire per round?')
    # parser.add_argument('--n_labels_pretraining', type=int, default=-1, help='How many labels to acquire before main training loop begins? Determines no. agent steps in pretraining. If -1 (default), it will be set to n_labels_per_round') # Ibarz: 25k. Removed support for diff no. labels in pretraining
    # parser.add_argument('--n_labels_per_round', type=int, nargs='+', default=[5]*20, help='How many labels to acquire per round? (in main training loop). len should be same as n_rounds')
    parser.add_argument('--batch_size_acq', type=int, default=1, help='In acquiring `n_labels_per_round`, what batch size are these acquired in? Reward model is trained after every acquisition batch. `batch_size_acq` == `n_labels_per_round`, is used in Christiano/Ibarz')
    parser.add_argument('--n_agent_steps', type=int, default=3000, help='No. of steps that agent takes per round in environment, while training every agent_gdt_step_period steps') # Ibarz: 100k
    parser.add_argument('--agent_test_frequency', type=int, default=1, help='Over the course of its n_agent_[train|total]_steps, how many times is agent performance tested? (and the run terminated if `terminate_once_solved == True`')
    parser.add_argument('--agent_learning_starts', type=int, default=0, help='After how many steps does the agent start making learning updates? This replaced the functionality of n_agent_total_steps.')
    parser.add_argument('--no_reinit_agent', dest='reinit_agent', action='store_false', help='Flag not to reinitialise the agent before every training round')
    # parser.add_argument('--period_half_lr', type=int, default=1750) # lr is halved every period_half_lr optimizer steps

    # reward model hyperparamas
    parser.add_argument('--hid_units_rm', type=int, default=64)
    parser.add_argument('--batch_size_rm', type=int, default=16) # same as Ibarz
    parser.add_argument('--lr_rm', type=float, default=1e-4)
    parser.add_argument('--p_dropout_rm', type=float, default=0.2)
    parser.add_argument('--lambda_rm', type=float, default=1e-4, help='coefficient for L2 regularization for reward_model optimization')
    parser.add_argument('--n_epochs_pretrain_rm', type=int, default=2000) # Ibarz: 50e3
    parser.add_argument('--n_epochs_train_rm', type=int, default=2000, help='No. epochs to train reward model per round in main training loop') # Ibarz: 6250
    parser.add_argument('--prefs_buffer_size', type=int, default=5000) # Ibarz: 6800. since currently we collect 400 labels in total currently, this doesn't matter (Ibarz never throw away labels. Christiano does.)
    parser.add_argument('--clip_length', type=int, default=25) # as per Ibarz/Christiano; i'm interested in changing this
    parser.add_argument('--force_label_choice', action='store_true', help='Does synthetic annotator label clips about which it is indifferent as 0.5? If `True`, label equally good clips randomly')
    parser.add_argument('--corr_rollout_steps', type=int, default=1000, help='When collecting rollouts to evaluate correlation of true and predicted reward, how many steps per rollout?')
    parser.add_argument('--corr_num_rollouts', type=int, default=5, help='When collecting rollouts to evaluate correlation of true and predicted reward, how many rollouts in total?')
    parser.add_argument('--no_ensemble_for_reward_pred', action='store_true', help='If true, then use ensemble for uncertainty estimates but pick a random net to compute rewards sent to DQN')
    parser.add_argument('--no_reinit_rm', dest='reinit_rm', action='store_false', help='Flag not to reinitialise reward model before every training round')
    parser.add_argument('--no_normalise_rm_while_training', action='store_true', help='Flag to not normalise output of reward predictors while training them (only while testing)')

    # active learning
    parser.add_argument('--active_method', type=str, default=None, help='Choice of: BALD, var_ratios, max_entropy, mean_std')
    parser.add_argument('--uncert_method', type=str, default=None, help='Choice of: MC, ensemble')
    parser.add_argument('--num_MC_samples', type=int, default=10)
    parser.add_argument('--acq_search_strategy', type=str, default='christiano', help='Whether to use christiano or all_pairs strategy to search for clip pairs. `angelos` is deprecated')
    parser.add_argument('--size_rm_ensemble', type=int, default=1, help='If active_method == ensemble then this must be >= 2')
    parser.add_argument('--selection_factor', type=int, default=10, help='when doing active learning, 1/selection_factor of the randomly sampled clip pairs are sent to human for evaluation')
    args = parser.parse_args()
    args = make_arg_changes(args)
    return args

def make_arg_changes(args):
    """Modifies or adds some experiment
       settings to args, and returns them.
    """
    if args.default_settings:
        default_args_map = {
            'cartpole': cartpole_defaults,
            'acrobot_sam': acrobot_sam_defaults,
            'openai': openai_defaults,
            'openai_atari': openai_atari_defaults
        }
        args = default_args_map[args.default_settings](args)

    assert args.n_labels_per_round % args.batch_size_acq == 0,\
        "Acquisition batch size is {}, but it should divide n_labels_per_round, which is {}".format(
            args.batch_size_acq, args.n_labels_per_round)
    args.n_acq_batches_per_round = args.n_labels_per_round // args.batch_size_acq

    args.exploration = LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.n_agent_steps),
                                      initial_p=args.epsilon_start,
                                      final_p=args.epsilon_stop)
    
    envs_to_ids = { 'cartpole': {'id': 'gym_barm:CartPoleContinual-v0',
                                'id_test': 'CartPole-v0',
                                },
                    'acrobot': {'id': 'Acrobot-v1', # standard Acrobot already has suitable reward function for casting as continuing task
                                'id_test': 'Acrobot-v1',
                               },
                    'acrobot_hard': {'id': 'gym_barm:AcrobotHard-v1', # standard Acrobot already has suitable reward function for casting as continuing task
                                'id_test': 'gym_barm:AcrobotHard-v1',
                               },
                    'mountain_car': {'id': 'gym_barm:MountainCarContinual-v0',
                                    'id_test': 'MountainCar-v0',
                                    },
                    'mountain_car_enriched': {'id': 'gym_barm:MountainCarContinualEnriched-v0',
                                              'id_test': 'MountainCar-v0', # test in standard env so we know how well agent is truly doing
                                             },
                    'cartpole_old': {'id': 'gym_barm:CartPole_Cont-v0',
                                'id_test': 'CartPole-v0',
                                },
                    'cartpole_old_rich': {'id': 'gym_barm:CartPole_EnrichedCont-v0',
                                      'id_test': 'gym_barm:CartPole_Enriched-v0',
                                     },
                    'pong': {'id': 'PongNoFrameskip-v0',
                             'id_test': 'PongNoFrameskip-v0',
                                     }
    }
    args.env_ID = envs_to_ids[args.env_str]['id']
    args.env_ID_test = envs_to_ids[args.env_str]['id_test']
    
    # check some things about RL training
    assert args.n_agent_steps % args.clip_length == 0,\
        "clip_length ({}) should be a factor of n_agent_steps ({})".format(
        args.clip_length, args.n_agent_steps)
    assert args.n_agent_steps % args.agent_test_frequency == 0,\
        "agent_test_frequency ({}} should be a factor of n_agent_steps ({})".format(
            args.agent_test_frequency, args.n_agent_steps)
    args.n_agent_steps_before_test = args.n_agent_steps // args.agent_test_frequency

    if args.test:
        args.n_runs = 1
        args.n_rounds = 1
        args.n_agent_steps = 1200
        args.n_epochs_pretrain_rm = 10
        args.n_epochs_train_rm = 10
        args.selection_factor = 2
    if args.RL_baseline:
        args.n_epochs_pretrain_rm = 0
        args.n_epochs_train_rm = 0
    if args.uncert_method == 'ensemble':
        assert args.size_rm_ensemble >= 2
    return args
    

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

    # make environment
    env = gym.make(args.env_ID)
    # if isinstance(env.env, gym.envs.atari.AtariEnv):
    if args.env_str == 'pong':
        env = preprocess_atari_env(env)
        args.dqn_archi = 'conv'
    else:
        args.dqn_archi = 'mlp'
    env.seed(args.random_seed)
    args.obs_shape = env.observation_space.shape[0] # env.observation_space is Box(4,) and calling .shape returns (4,) [gym can be ugly]
    args.obs_shape_all = env.observation_space.shape # TODO ugly
    assert isinstance(env.action_space, gym.spaces.Discrete), 'DQN requires discrete action space.'
    args.act_shape = 1 # [gym doesn't have a nice way to get shape of Discrete space... env.action_space.shape -> () ]
    args.obs_act_shape = args.obs_shape + args.act_shape
    args.n_actions = env.action_space.n

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
            logging.exception('RUN {}/{} FAILED with the following traceback:\n'.format(i_run, args.n_runs))

if __name__ == '__main__':
    main()