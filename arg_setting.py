import argparse
import numpy as np
from defaults import *
from q_learning import LinearSchedule

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--info', type=str, default='', help='Tensorboard log is saved in ./logs/i_run/random_seed/[true|pred]/')
    parser.add_argument('--default_settings', type=str, default=None, help='Flag to override args with those in default.py. Choice of: acrobot_sam, openai, openai_atari, cartpole, gridworld')
    parser.add_argument('--env_str', type=str, default=None, help='Choice of: acrobot, mountain_car, cartpole, cartpole_old, cartpole_old_rich, frozen_lake, gridworld')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs to repeat the experiment')
    parser.add_argument('--n_rounds', type=int, default=5, help='number of rounds to repeat main training loop')
    parser.add_argument('--RL_baseline', action='store_true', help='Do RL baseline instead of reward learning?')
    parser.add_argument('--random_policy', action='store_true', help='Do the experiments with an entirely random policy, to benchmark performance')
    parser.add_argument('--test', action='store_true', help='Flag to make training procedure very short (to check for errors)')
    parser.add_argument('--save_video', action='store_true', help='Flag to save as .mp4 some test episodes and final ~3 train episodes')
    parser.add_argument('--n_test_vids_to_save', type=int, default=-1, help='If save_video is true, this specifies how many of the 100 test episodes to save as .mp4. Default -1 means all episodes will be saved')
    parser.add_argument('--save_pair_videos', action='store_true', help='Flag to save videos of acquired clip pairs')
    parser.add_argument('--continue_once_solved', action='store_true', help='Experiment will continue even when agent test mean ep return >= env.spec.reward_threshold')
    parser.add_argument('--seed_offset', type=int, default=0, help='We seed with i_run + seed_offset, where i_run in {0..n_runs-1}')
    parser.add_argument('--n_sample_reps', type=int, default=1, help='For debugging: if >1, this will cause n_sample_reps exact copies of the first clip sampled from AgentExperience to be given to acquisition function')
    parser.add_argument('--reinit_rm_when_q_learning', action='store_true', help='For debugging: this will do the crazy thing of reinitialising reward model every time we want to use it to send rewards to DQN')

    # agent hyperparams
    parser.add_argument('--dqn_archi', type=str, default='mlp', help='Is deep Q-network an mlp or cnn?')
    parser.add_argument('--dqn_loss', type=str, default='mse', help='Use mse or huber loss function?')
    parser.add_argument('--h1_agent', type=int, default=32)
    parser.add_argument('--h2_agent', type=int, default=64)
    parser.add_argument('--batch_size_agent', type=int, default=32)
    parser.add_argument('--lr_agent', type=float, default=1e-3)
    parser.add_argument('--lambda_agent', type=float, default=1e-4, help='coefficient for L2 regularization for agent optimization')
    parser.add_argument('--replay_buffer_size', type=int, default=30000)
    parser.add_argument('--target_update_period', type=int, default=1) # Ibarz: 8000, but hard updates
    parser.add_argument('--target_update_tau', type=float, default=8e-2) # Ibarz: 1 (hard update)
    parser.add_argument('--agent_gdt_step_period', type=int, default=4) # Ibarz: 4
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='exploration probability for agent at start')
    # parser.add_argument('--epsilon_decay', type=float, default=0.999, help='`epsilon *= epsilon * epsilon_decay` every learning step, until `epsilon_stop`') 
    parser.add_argument('--epsilon_stop', type=float, default=0.01)
    parser.add_argument('--exploration_fraction', type=float, default=0.1, help='Over what fraction of entire training period is epsilon annealed (linearly)?')
    # parser.add_argument('--n_labels_per_round', type=int, default=5, help='How many labels to acquire per round?')
    # parser.add_argument('--n_labels_pretraining', type=int, default=500, help='How many labels to acquire before main training loop begins? Determines no. agent steps in pretraining. If -1 (default), it will be set to n_labels_per_round') # Ibarz: 25k. Removed support for diff no. labels in pretraining
    parser.add_argument('--n_labels_per_round', type=int, nargs='+', default=[500,3000,1500,750,750,500], help='How many labels to acquire per round? (in main training loop). len should be n_rounds + 1, since 0th is pretraining labels')
    parser.add_argument('--batch_size_acq', type=int, default=-1, help='In acquiring `n_labels_per_round`, what batch size are these acquired in? Reward model is trained after every acquisition batch. If -1 (default), `batch_size_acq` == `n_labels_per_round`, as in Christiano/Ibarz')
    parser.add_argument('--n_agent_steps', type=int, default=150000, help='No. of steps that agent takes per round in environment, while training every agent_gdt_step_period steps') # Ibarz: 100k
    parser.add_argument('--n_agent_steps_pretrain', type=int, default=-1, help='No. of steps that agent takes before main training loop begins. epsilon=0.5 for these steps. If -1 (default) then n_agent_steps_pretrain will be determined by n_labels_per_round (will collect just enough)')
    parser.add_argument('--agent_test_frequency', type=int, default=15, help='Over the course of its n_agent_[train|total]_steps, how many times is agent performance tested? (and the run terminated if `terminate_once_solved == True`')
    parser.add_argument('--agent_learning_starts', type=int, default=0, help='After how many steps does the agent start making learning updates? This replaced the functionality of n_agent_total_steps.')
    parser.add_argument('--no_reinit_agent', dest='reinit_agent', action='store_false', help='Flag not to reinitialise the agent before every training round')
    parser.add_argument('--no_normalise_rewards', dest='normalise_rewards', action='store_false', help='Flag not to normalise rewards sent to the agent (either true or predicted, depending on args.RL_baseline)')
    parser.add_argument('--agent_gets_dones', action='store_true', help='Flag to store done=True signals in replay buffer (Christiano/Ibarz say not to do this, but we want to see how it affects performance)')
    # parser.add_argument('--period_half_lr', type=int, default=1750) # lr is halved every period_half_lr optimizer steps

    # reward model hyperparamas
    parser.add_argument('--rm_archi', type=str, default='mlp', help='Is reward model an mlp, cnn or cnn_mod?')
    parser.add_argument('--hid_units_rm', type=int, default=64)
    parser.add_argument('--batch_size_rm', type=int, default=16) # same as Ibarz
    parser.add_argument('--lr_rm', type=float, default=1e-4)
    parser.add_argument('--p_dropout_rm', type=float, default=0.5)
    parser.add_argument('--lambda_rm', type=float, default=1e-4, help='coefficient for L2 regularization for reward_model optimization')
    parser.add_argument('--n_epochs_pretrain_rm', type=int, default=-1, help='No. epochs to train rm on preferences collected during initial rollouts. If -1 (default) then this will be set to n_epochs_train_rm') # Ibarz: 50e3
    parser.add_argument('--n_epochs_train_rm', type=int, default=3000, help='No. epochs to train reward model per round in main training loop') # Ibarz: 6250
    # parser.add_argument('--prefs_buffer_size', type=int, default=1000) # Ibarz: 6800. since currently we collect fewer than 1000 labels in total, this doesn't matter (Ibarz never throw away labels. Christiano does.)
    # NB using 5000 with obs_act_shape of (21168,) gives MemoryError. So if I do need to increase its size much more, I may need to change the implementation somehow...
    parser.add_argument('--clip_length', type=int, default=25) # as per Ibarz/Christiano; i'm interested in changing this
    parser.add_argument('--force_label_choice', action='store_true', help='Does synthetic annotator label clips about which it is indifferent as 0.5? If `True`, label equally good clips randomly')
    parser.add_argument('--corr_rollout_steps', type=int, default=1000, help='When collecting rollouts to evaluate correlation of true and predicted reward, how many steps per rollout?')
    parser.add_argument('--corr_num_rollouts', type=int, default=5, help='When collecting rollouts to evaluate correlation of true and predicted reward, how many rollouts in total?')
    parser.add_argument('--no_ensemble_for_reward_pred', action='store_true', help='If true, then use ensemble for uncertainty estimates but pick a random net to compute rewards sent to DQN')
    parser.add_argument('--no_reinit_rm', dest='reinit_rm', action='store_false', help='Flag not to reinitialise reward model before every training round')
    parser.add_argument('--normalise_rm_while_training', action='store_true', help='Flag to normalise output of reward predictors while training them (only while testing)')
    parser.add_argument('--no_train_reward_model', action='store_true', help='Flag not train reward model. Sets lr_rm to zero')

    # active learning
    parser.add_argument('--active_method', type=str, default=None, help='Choice of: BALD, var_ratios, max_entropy, mean_std')
    parser.add_argument('--uncert_method', type=str, default=None, help='Choice of: MC, ensemble')
    parser.add_argument('--num_MC_samples', type=int, default=10)
    parser.add_argument('--acq_search_strategy', type=str, default='christiano', help='Whether to use christiano or all_pairs strategy to search for clip pairs. `angelos` is deprecated')
    parser.add_argument('--size_rm_ensemble', type=int, default=1, help='If active_method == ensemble then this must be >= 2')
    parser.add_argument('--selection_factor', type=int, default=10, help='when doing active learning, 1/selection_factor of the randomly sampled clip pairs are sent to human for evaluation')

    # settings that apply only to gridworld
    parser.add_argument('--grid_partial_obs', action='store_true', help='Is environment only partially observable to agent?')
    parser.add_argument('--grid_size', type=int, default=5, help='Length and width of grid')
    parser.add_argument('--grid_deterministic_reset', action='store_true', help='Do objects in grid reset to same positions once episode terminates?')
    parser.add_argument('--grid_no_terminate_ep_if_done', action='store_true', help='Flag to make env.step() not give done=True when agent reaches goal or lava, but only when env.spec.max_episode_steps is reached.')
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
            'openai_atari': openai_atari_defaults,
            'gridworld_nb': gridworld_nb_defaults,
            'gridworld_zac': gridworld_zac_defaults
        }
        args = default_args_map[args.default_settings](args)

    if len(args.n_labels_per_round) == 1:
        args.n_labels_per_round = [args.n_labels_per_round] * (args.n_rounds + 1)

    assert len(args.n_labels_per_round) == args.n_rounds + 1

    if args.batch_size_acq == -1:
        args.batch_size_acq = args.n_labels_per_round
    
    assert (np.array(args.n_labels_per_round) % np.array(args.batch_size_acq) == 0).all(),\
        "Acquisition batch size is {}, but it should divide n_labels_per_round, which is {}".format(
            args.batch_size_acq, args.n_labels_per_round)
            
    args.n_acq_batches_per_round = np.array(args.n_labels_per_round) // np.array(args.batch_size_acq)

    args.exploration = LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.n_agent_steps),
                                      initial_p=args.epsilon_start,
                                      final_p=args.epsilon_stop)

    args.prefs_buffer_size = sum(args.n_labels_per_round)
    
    # get environment ID
    envs_to_ids = { 'cartpole': {'id': 'gym_barm:CartPoleContinual-v0',
                                'id_test': 'CartPole-v0',
                                },
                    'cartpole_scrambled': {'id': 'gym_barm:CartPoleScrambled-v0',
                                           'id_test': 'CartPole-v0', # test in standard env so we know how well agent is truly doing
                                },
                    'acrobot': {'id': 'Acrobot-v1', # standard Acrobot already has suitable reward function for casting as continuing task
                                'id_test': 'Acrobot-v1',
                               },
                    'acrobot_hard': {'id': 'gym_barm:AcrobotHard-v1',
                                     'id_test': 'gym_barm:AcrobotHard-v1',
                               },
                    'acrobot_scrambled': {'id': 'gym_barm:AcrobotScrambled-v1',
                                     'id_test': 'Acrobot-v1', # test in standard env so we know how well agent is truly doing
                               },
                    'acrobot_all_scrambled': {'id': 'gym_barm:AcrobotScrambled-v1',
                                     'id_test': 'gym_barm:AcrobotScrambled-v1', # this test will be meaningless
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
                                     },
                    'frozen_lake': {'id': 'FrozenLake-v0',
                                    'id_test': 'FrozenLake-v0',
                                     },
                    'gridworld': {'id': 'Gridworld-v0',
                                  'id_test': 'Gridworld-v0',
                                 }
    }
    try:
        args.env_ID = envs_to_ids[args.env_str]['id']
        args.env_ID_test = envs_to_ids[args.env_str]['id_test']
    except KeyError:
        raise KeyError("You specified {} as the env_str. I don't know what that is!".format(args.env_str))
    
    # get environment kwargs for gridworld
    args.env_kwargs = {}
    if args.env_str == 'gridworld':
        args.env_kwargs['partial']              = args.grid_partial_obs
        args.env_kwargs['size']                 = args.grid_size
        args.env_kwargs['random_resets']        = not args.grid_deterministic_reset
        args.env_kwargs['terminate_ep_if_done'] = not args.grid_no_terminate_ep_if_done

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
        args.n_agent_steps = 5000
        args.n_epochs_pretrain_rm = 10
        args.n_epochs_train_rm = 10
        args.selection_factor = 2
    if args.n_epochs_pretrain_rm == -1:
        args.n_epochs_pretrain_rm = args.n_epochs_train_rm
    if args.RL_baseline or args.no_train_reward_model:
        args.n_epochs_pretrain_rm = 0
        args.n_epochs_train_rm    = 0
    if args.uncert_method == 'ensemble':
        assert args.size_rm_ensemble >= 2
    return args