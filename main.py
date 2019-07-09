import math, random, argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from q_learning import *
from reward_model import *
from active_learning import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--env_class', type=str, default='CartPoleContinuous-v0')
    parser.add_argument('--n_rounds', type=int, default=10, help='number of rounds to repeat main training loop')
    parser.add_argument('--RL_baseline', action='store_true', help='Do RL baseline instead of reward learning?')
    parser.add_argument('--random_policy', action='store_true', help='Do the experiments with an entirely random policy, to benchmark performance')
    parser.add_argument('--ep_end_penalty', type=float, default=-29.0, help='How much reward does agent get when the (dummy) episode ends?')
    parser.add_argument('--info', type=str, default='', help='Tensorboard log is saved in ./logs/*info*_pred/true')
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
    parser.add_argument('--n_agent_steps', type=int, default=2000, help='No. of steps that agent takes in environment, per round (in main training loop)') # Ibarz: 100k
    parser.add_argument('--dummy_ep_length', type=int, default=200, help="After how many steps do we interpret an 'episode' as having elapsed and log performance? (This affects only result presentation not algo)")
    # parser.add_argument('--period_half_lr', type=int, default=1750) # lr is halved every period_half_lr optimizer steps

    # reward model hyperparamas
    parser.add_argument('--hid_units_rm', type=int, default=64)
    parser.add_argument('--batch_size_rm', type=int, default=16) # same as Ibarz
    parser.add_argument('--lr_rm', type=float, default=1e-4)
    parser.add_argument('--p_dropout_rm', type=float, default=0.2)
    parser.add_argument('--lambda_rm', type=float, default=1e-4, help='coefficient for L2 regularization for reward_model optimization')
    parser.add_argument('--n_epochs_pretrain_rm', type=int, default=1000) # Ibarz uses 50e3 ... but this gave me NaNs so I'm scared...
    parser.add_argument('--n_epochs_train_rm', type=int, default=1000, help='No. epochs to train reward model per round in main training loop') # Ibarz: 6250
    parser.add_argument('--prefs_buffer_size', type=int, default=1000) # Ibarz: 6800. since currently we collect strictly lt 100 + 50*5 = 350 labels this doesn't matter
    parser.add_argument('--clip_length', type=int, default=25) # as per Ibarz/Christiano; i'm interested in changing this
    parser.add_argument('--force_label_choice', action='store_true', help='Does synthetic annotator label clips about which it is indifferent as 0.5? If `True`, label equally good clips randomly')
    parser.add_argument('--corr_rollout_steps', type=int, default=1000, help='When collecting rollouts to evaluate correlation of true and predicted reward, how many steps per rollout?')
    parser.add_argument('--corr_num_rollouts', type=int, default=5, help='When collecting rollouts to evaluate correlation of true and predicted reward, how many rollouts in total?')

    # active learning
    parser.add_argument('--active_learning', type=str, default=None, help='Choice of: MC_variance, info_gain, ensemble_variance')
    parser.add_argument('--num_MC_samples', type=int, default=10)
    parser.add_argument('--acquisition_search_strategy', type=str, default='v1', help='Whether to use Christiano (v0) or Angelos (v1) strategy to search for clip pairs')
    parser.add_argument('--size_rm_ensemble', type=int, default=1, help='If active_learning == ensemble_variance then this must be >= 2')
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
        args.n_rounds=1
        # args.n_initial_agent_steps=3000
        # args.n_agent_steps=3000
        args.n_epochs_pretrain_rm=10
        args.n_epochs_train_rm=10
    if args.active_learning == 'ensemble_variance':
        assert args.size_rm_ensemble >= 2
    return args
    
def do_random_experiment(env, args, writer1, writer2):
    for i_train_round in range(args.n_rounds):
        print('[Start Round {}]'.format(i_train_round))
        dummy_returns = {'ep': 0, 'all': []}
        env.reset()
        for step in trange(args.n_agent_steps, desc='Taking random actions for {} steps'.format(args.n_agent_steps), dynamic_ncols=True):
            # agent interact with env
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            _, r_true, _, _ = env.step(action) # one continuous episode
            dummy_returns['ep'] += r_true # record step info

            # log performance after a "dummy" episode has elapsed
            if (step % args.dummy_ep_length == 0 or step == args.n_agent_steps - 1):
                writer2.add_scalar('2.dummy ep return against step/round {}'.format(i_train_round), dummy_returns['ep'], step)
                dummy_returns['all'].append(dummy_returns['ep'])
                dummy_returns['ep'] = 0

        # log mean recent return this training round
        # mean_dummy_true_returns = np.sum(np.array(dummy_returns['all'][-3:])) / 3. # 3 dummy eps is the final 3*200/2000 == 3/10 eps in the round
        mean_dummy_true_returns = np.sum(np.array(dummy_returns['all']))
        writer2.add_scalar('1.mean dummy ep returns per training round', mean_dummy_true_returns, i_train_round)

def do_pretraining(env, q_net, reward_model, prefs_buffer, args, obs_shape, act_shape, writer1, writer2):
    # Stage 0.1 Initialise policy and do some rollouts
    epsilon_pretrain = 0.1 # for now I'll use a constant epilson during pretraining
    # n_initial_steps = args.n_initial_agent_steps
    n_initial_steps = args.n_labels_pretraining * 2 * args.clip_length
    if args.active_learning:
        n_initial_steps *= args.selection_factor
        print('Doing Active Learning ({} method), so collect {}x more rollouts than usual'.format(
                args.active_learning, args.selection_factor))
    num_clips = int(n_initial_steps//args.clip_length)
    assert n_initial_steps % args.clip_length == 0, "Agent should take a number of steps that's divisible by the desired clip_length"
    agent_experience = AgentExperience((num_clips, args.clip_length, obs_shape+act_shape), args.force_label_choice)
    state = env.reset()
    for _ in trange(n_initial_steps, desc='Stage 0.1: Collecting rollouts from untrained policy, {} agent steps'.format(n_initial_steps), dynamic_ncols=True):
        action = q_net.act(state, epsilon_pretrain)
        assert env.action_space.contains(action)
        next_state, r_true, _, _ = env.step(action)    
        # record step information
        sa_pair = torch.tensor(np.append(state, action)).float()
        agent_experience.add(sa_pair, r_true) # add reward too in order to produce synthetic prefs
        state = next_state

    # Stage 0.2 Sample without replacement from those rollouts and label them (synthetically)
    # TODO abstract this and use the same function in training
    # num_pretraining_labels = args.n_initial_agent_steps // (2*args.clip_length)
    print('Stage 0.2: Sample without replacement from those rollouts to collect {} labels. Each label is on a pair of clips of length {}'.format(args.n_labels_pretraining, args.clip_length))
    writer1.add_scalar('6.labels requested per round', args.n_labels_pretraining, -1)
    if args.active_learning:
        if args.acquisition_search_strategy == 'v0':
                clip_pairs, rews, mus = acquire_clip_pairs_v0(agent_experience, reward_model, args.n_labels_pretraining, args, writer1, writer2, i_train_round=-1)
        elif args.acquisition_search_strategy == 'v1':
            clip_pairs, rews, mus = acquire_clip_pairs_v1(agent_experience, reward_model, args.n_labels_pretraining, args, writer1, writer2, i_train_round=-1)
    else:
        clip_pairs, rews, mus = agent_experience.sample_pairs(args.n_labels_pretraining)
    # put chosen clip_pairs, true rewards (just to compute mean/var of true reward across prefs_buffer)
    # and synthetic preferences into prefs_buffer
    prefs_buffer.push(clip_pairs, rews, mus)
    
    # Stage 0.3 Intialise and pretrain reward model
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm)
    reward_model.train() # dropout on
    for epoch in trange(args.n_epochs_pretrain_rm, desc='Stage 0.3: Intialise and pretrain reward model for {} batches on those preferences'.format(args.n_epochs_pretrain_rm), dynamic_ncols=True):
        with torch.autograd.detect_anomaly(): # detects NaNs; useful for debugging
            clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
            r_hats_batch = reward_model(clip_pair_batch).squeeze(-1)
            loss_rm = compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, obs_shape, act_shape)
            # TODO call clean version instead i.e. loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
            reward_model.train() # dropout
            optimizer_rm.zero_grad()
            loss_rm.backward()
            optimizer_rm.step()
            writer1.add_scalar('7.reward model loss/pretraining', loss_rm, epoch)

    # evaluate reward model correlation after pretraining
    if not args.RL_baseline:
        print('Reward model training complete... Evaluating reward model correlation on {} state-action pairs, accumulated on {} rollouts of length {}'.format(
                args.corr_rollout_steps * args.corr_num_rollouts, args.corr_num_rollouts, args.corr_rollout_steps))
        r_xy, plots = eval_rm_correlation(reward_model, env, q_net, args, obs_shape, act_shape, rollout_steps=args.corr_rollout_steps, num_rollouts=args.corr_num_rollouts)
        log_correlation(r_xy, plots, writer1, round_num=-1)

    return reward_model, prefs_buffer


def do_training(env, q_net, q_target, reward_model, prefs_buffer, args, obs_shape, act_shape, writer1, writer2):
    # Stage 1.0: Setup
    optimizer_agent = optim.Adam(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent) # q_net initialised above
    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm) # reinitialise optimizer so we don't need to pass it between funcs

    for i_train_round in range(args.n_rounds):
        print('[Start Round {}]'.format(i_train_round))
        # Stage 1.1: Reinforcement learning with (normalised) rewards from current reward model
        q_net, q_target, replay_buffer, agent_experience = do_RL(env, q_net, q_target, optimizer_agent, replay_buffer,
                                                                 reward_model, prefs_buffer, args, i_train_round,
                                                                 obs_shape, act_shape, writer1, writer2)
        
        # Stage 1.2: Sample clip pairs without replacement from recent rollouts and label them (synthetically)
        # num_labels_requested = int(50*5 / (i_train_round + 5)) #int(58.56 * (5e6 / (i_train_round * args.n_agent_steps + 5e6) )) # compute_label_annealing_const.py
        num_labels_requested = args.n_labels_per_round[i_train_round]
        print('Stage 1.2: Sample without replacement from those rollouts to collect {} labels/preference tuples'.format(num_labels_requested))
        writer1.add_scalar('6.labels requested per round', num_labels_requested, i_train_round)
        if args.active_learning:
            if args.acquisition_search_strategy == 'v0':
                clip_pairs, rews, mus = acquire_clip_pairs_v0(agent_experience, reward_model, num_labels_requested, args, writer1, writer2, i_train_round)
            elif args.acquisition_search_strategy == 'v1':
                clip_pairs, rews, mus = acquire_clip_pairs_v1(agent_experience, reward_model, num_labels_requested, args, writer1, writer2, i_train_round)
        else:
            clip_pairs, rews, mus = agent_experience.sample_pairs(num_labels_requested)
        # put labelled clip_pairs into prefs_buffer
        assert len(clip_pairs) == num_labels_requested
        prefs_buffer.push(clip_pairs, rews, mus)
        
        # Stage 1.3: Train reward model
        reward_model.train() # dropout on
        for epoch in trange(args.n_epochs_train_rm, desc='Stage 1.3: Train reward model for {} batches on those preferences'.format(args.n_epochs_train_rm), dynamic_ncols=True):
            with torch.autograd.detect_anomaly():
                clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
                r_hats_batch = reward_model(clip_pair_batch).squeeze(-1) # squeeze the oa_pair dimension that was passed through reward_model
                assert clip_pair_batch.shape == (args.batch_size_rm, 2, args.clip_length, obs_shape + act_shape)
                loss_rm = compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, obs_shape, act_shape)
                # TODO call clean version instead i.e. loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
                optimizer_rm.zero_grad()
                loss_rm.backward()
                optimizer_rm.step()
                writer1.add_scalar('7.reward model loss/round {}'.format(i_train_round), loss_rm, epoch)

        # evaluate reward model correlation
        if not args.RL_baseline:
            print('Reward model training complete... Evaluating reward model correlation on {} state-action pairs, accumulated on {} rollouts of length {}'.format(
                args.corr_rollout_steps * args.corr_num_rollouts, args.corr_num_rollouts, args.corr_rollout_steps))
            r_xy, plots = eval_rm_correlation(reward_model, env, q_net, args, obs_shape, act_shape, rollout_steps=args.corr_rollout_steps, num_rollouts=args.corr_num_rollouts)
            log_correlation(r_xy, plots, writer1, round_num=i_train_round)


def main(): 
    # experiment settings
    args = parse_arguments()
    print('\nRunning experiment with the following settings:')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # for reproducibility
    torch.manual_seed(args.random_seed) # TODO check that setting random seed here also applies to random calls in modules
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # TensorBoard logging
    logdir = './logs/test/' if args.test else './logs/'
    writer1 = SummaryWriter(log_dir=logdir+'{}_pred'.format(args.info))
    writer2 = SummaryWriter(log_dir=logdir+'{}_true'.format(args.info))

    # make environment
    env = gym.make(args.env_class, ep_end_penalty=args.ep_end_penalty)
    obs_shape = env.observation_space.shape[0] # env.observation_space is Box(4,) and calling .shape returns (4,) [gym can be ugly]
    assert isinstance(env.action_space, gym.spaces.Discrete), 'DQN requires discrete action space.'
    act_shape = 1 # [gym doesn't have a nice way to get shape of Discrete space... env.action_space.shape -> () ]
    n_actions = env.action_space.n # env.action_space is Discrete(2) and calling .n returns 2
    args.obs_act_shape = obs_shape + act_shape

    if args.random_policy:
        do_random_experiment(env, args, writer1, writer2)
    else:
        # instantiate neural nets and buffer for preferences
        q_net = DQN(obs_shape, n_actions, args)
        q_target = DQN(obs_shape, n_actions, args)
        q_target.load_state_dict(q_net.state_dict()) # set params of q_target to be the same
        if args.size_rm_ensemble >= 2:
            reward_model = RewardModelEnsemble(obs_shape, act_shape, args)
            print('Using a {}-ensemble of nets for our reward model'.format(args.size_rm_ensemble))
        else:
            reward_model = RewardModel(obs_shape, act_shape, args)
        prefs_buffer = PrefsBuffer(capacity=args.prefs_buffer_size, clip_shape=(args.clip_length, obs_shape+act_shape))
        # if args.active_learning == 'BALD': # TODO better code design to instantiate acq_funcs out here? do we need different acq_funcs for singles and pairs?
        #     acq_func = 'blah'
        # fire away!
        reward_model, prefs_buffer = do_pretraining(env, q_net, reward_model, prefs_buffer, args, obs_shape, act_shape, writer1, writer2)
        do_training(env, q_net, q_target, reward_model, prefs_buffer, args, obs_shape, act_shape, writer1, writer2)
    
    writer1.close()
    writer2.close()

if __name__ == '__main__':
    main()