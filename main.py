import random, argparse, logging
import numpy as np
import pandas as pd
import gym, gym_barm
import torch
from torch.utils.tensorboard import SummaryWriter

from training_protocol import *
from q_learning import *
from reward_learning import *
from active_learning import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--info', type=str, default='', help='Tensorboard log is saved in ./logs/i_run/random_seed/[true|pred]/')
    parser.add_argument('--env_class', type=str, default='gym_barm:CartPoleContinuous-v0')
    parser.add_argument('--env_class_test', type=str, default='CartPole-v0', help='We use the standard, non-continuous version of the env for testing agent performance')
    parser.add_argument('--n_runs', type=int, default=20, help='number of runs to repeat the experiment')
    parser.add_argument('--n_rounds', type=int, default=20, help='number of rounds to repeat main training loop')
    parser.add_argument('--RL_baseline', action='store_true', help='Do RL baseline instead of reward learning?')
    parser.add_argument('--random_policy', action='store_true', help='Do the experiments with an entirely random policy, to benchmark performance')
    parser.add_argument('--ep_end_penalty', type=float, default=-29.0, help='How much reward does agent get when the (dummy) episode ends?')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--test', action='store_true', help='Flag to make training procedure very short (to check for errors)')
    # agent hyperparams
    parser.add_argument('--h1_agent', type=int, default=32)
    parser.add_argument('--h2_agent', type=int, default=64)
    parser.add_argument('--batch_size_agent', type=int, default=32)
    parser.add_argument('--lr_agent', type=float, default=1e-3)
    parser.add_argument('--lambda_agent', type=float, default=1e-4, help='coefficient for L2 regularization for agent optimization')
    parser.add_argument('--replay_buffer_size', type=int, default=int(1e5))
    parser.add_argument('--target_update_period', type=int, default=1) # Ibarz: 8000, but hard updates
    parser.add_argument('--target_update_tau', type=float, default=8e-2) # Ibarz: 1 (hard update)
    parser.add_argument('--agent_gdt_step_period', type=int, default=1) # Ibarz: 4
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='exploration probability for agent at start')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='`epsilon *= epsilon * epsilon_decay` every learning step, until `epsilon_stop`') 
    parser.add_argument('--epsilon_stop', type=float, default=0.01)
    parser.add_argument('--n_labels_pretraining', type=int, default=10, help='How many labels to acquire before main training loop begins? Determines no. agent steps in pretraining') # Ibarz: 25k
    parser.add_argument('--n_labels_per_round', type=int, nargs='+', default=[5]*10, help='How many labels to acquire per round? (in main training loop). len should be same as n_rounds')
    parser.add_argument('--n_agent_steps', type=int, default=3000, help='No. of steps that agent takes in environment, per round (in main training loop)') # Ibarz: 100k
    parser.add_argument('--dummy_ep_length', type=int, default=200, help="After how many steps do we interpret an 'episode' as having elapsed and log performance? (This affects only result presentation not algo)")
    # parser.add_argument('--period_half_lr', type=int, default=1750) # lr is halved every period_half_lr optimizer steps

    # reward model hyperparamas
    parser.add_argument('--hid_units_rm', type=int, default=64)
    parser.add_argument('--batch_size_rm', type=int, default=16) # same as Ibarz
    parser.add_argument('--lr_rm', type=float, default=1e-4)
    parser.add_argument('--p_dropout_rm', type=float, default=0.2)
    parser.add_argument('--lambda_rm', type=float, default=1e-4, help='coefficient for L2 regularization for reward_model optimization')
    parser.add_argument('--n_epochs_pretrain_rm', type=int, default=2000) # Ibarz: 50e3
    parser.add_argument('--n_epochs_train_rm', type=int, default=2000, help='No. epochs to train reward model per round in main training loop') # Ibarz: 6250
    parser.add_argument('--prefs_buffer_size', type=int, default=1000) # Ibarz: 6800. since currently we collect strictly lt 100 + 50*5 = 350 labels this doesn't matter
    parser.add_argument('--clip_length', type=int, default=25) # as per Ibarz/Christiano; i'm interested in changing this
    parser.add_argument('--force_label_choice', action='store_true', help='Does synthetic annotator label clips about which it is indifferent as 0.5? If `True`, label equally good clips randomly')
    parser.add_argument('--corr_rollout_steps', type=int, default=1000, help='When collecting rollouts to evaluate correlation of true and predicted reward, how many steps per rollout?')
    parser.add_argument('--corr_num_rollouts', type=int, default=5, help='When collecting rollouts to evaluate correlation of true and predicted reward, how many rollouts in total?')

    # active learning
    parser.add_argument('--active_method', type=str, default=None, help='Choice of: BALD, var_ratios, max_entropy, mean_std')
    parser.add_argument('--uncert_method', type=str, default=None, help='Choice of: MC, ensemble')
    parser.add_argument('--num_MC_samples', type=int, default=10)
    parser.add_argument('--acq_search_strategy', type=str, default='v0', help='Whether to use Christiano (v0) or Angelos (v1) strategy to search for clip pairs')
    parser.add_argument('--size_rm_ensemble', type=int, default=1, help='If active_method == ensemble then this must be >= 2')
    parser.add_argument('--selection_factor', type=int, default=10, help='when doing active learning, 1/selection_factor of the randomly sampled clip pairs are sent to human for evaluation')
    # if doing active learning n_steps_(pre)train is automatically increased by this factor bc we consider
    # sample complexity rather than computational complexity (we assume it's cheap for the agent to do rollouts
    # and we want to find whether active learning using the same amount of *data from the human* beats the random baseline)
    args = parser.parse_args()
    if args.RL_baseline:
        args.n_epochs_pretrain_rm = 0
        args.n_epochs_train_rm = 0
    else:
        assert len(args.n_labels_per_round) == args.n_rounds, "Experiment has {} rounds, but you specified the number labels to collect in {} rounds".format(args.n_rounds, len(args.n_labels_per_round))
    if args.test:
        args.n_runs = 1
        args.n_rounds = 2
        # args.n_initial_agent_steps=3000
        # args.n_agent_steps=3000
        args.n_epochs_pretrain_rm = 10
        args.n_epochs_train_rm = 10
    if args.uncert_method == 'ensemble':
        assert args.size_rm_ensemble >= 2
    return args
    

def run_experiment(args, i_run, returns_summary):
    # for reproducibility
    random_seed = i_run
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # TensorBoard logging
    logdir = './logs/{}/{}'.format(args.info, random_seed)
    writer1 = SummaryWriter(log_dir=logdir+'/true')
    writer2 = SummaryWriter(log_dir=logdir+'/pred')
    writer3 = SummaryWriter(log_dir=logdir+'/other')
    writers = [writer1, writer2, writer3]

    # make environment
    env = gym.make(args.env_class, ep_end_penalty=args.ep_end_penalty)
    obs_shape = env.observation_space.shape[0] # env.observation_space is Box(4,) and calling .shape returns (4,) [gym can be ugly]
    assert isinstance(env.action_space, gym.spaces.Discrete), 'DQN requires discrete action space.'
    act_shape = 1 # [gym doesn't have a nice way to get shape of Discrete space... env.action_space.shape -> () ]
    n_actions = env.action_space.n # env.action_space is Discrete(2) and calling .n returns 2
    args.obs_shape = obs_shape
    args.act_shape = act_shape
    args.obs_act_shape = obs_shape + act_shape

    if args.random_policy:
        do_random_experiment(env, args, returns_summary, writers, i_run)
    else:
        # instantiate neural nets and buffer for preferences
        q_net = DQN(obs_shape, n_actions, args)
        q_target = DQN(obs_shape, n_actions, args)
        q_target.load_state_dict(q_net.state_dict()) # set params of q_target to be the same
        if args.size_rm_ensemble >= 2:
            reward_model = RewardModelEnsemble(obs_shape, act_shape, args)
            logging.info('Using a {}-ensemble of nets for our reward model'.format(args.size_rm_ensemble))
        else:
            reward_model = RewardModel(obs_shape, act_shape, args)
        prefs_buffer = PrefsBuffer(capacity=args.prefs_buffer_size, clip_shape=(args.clip_length, obs_shape+act_shape))
        # fire away!
        reward_model, prefs_buffer = do_pretraining(env, q_net, reward_model, prefs_buffer, args, writers)
        do_training(env, q_net, q_target, reward_model, prefs_buffer, args, writers, returns_summary, i_run)
    
    writer1.close()
    writer2.close()
    writer3.close()

def main():
    args = parse_arguments()
    logging.basicConfig(filename='./logs/{}.log'.format(args.info), level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler()) # makes messages print to stderr, too
    logging.info('\nRunning experiment with the following settings:')
    for arg in vars(args):
        logging.info('{}: {}'.format(arg, getattr(args, arg)))
    
    returns_summary = {i: {} for i in range(args.n_runs)}
    for i_run in range(args.n_runs):
        try:
            run_experiment(args, i_run, returns_summary)
        except:
            logging.exception('Run {} of experiment {} failed')
    pd.DataFrame(returns_summary).to_csv('./logs/{}.csv'.format(args.info), index_label=['ep return type', 'round no.'])

if __name__ == '__main__':
    main()