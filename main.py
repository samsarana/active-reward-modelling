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
from tqdm import tqdm, trange

from q_learning import *
from reward_model import *
from active_learning import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--env_class', type=str, default='CartPoleContinuous-v0')
    parser.add_argument('--n_rounds', type=int, default=5, help='number of rounds to repeat main training loop')
    parser.add_argument('--RL_baseline', type=bool, default=False, help='do RL baseline instead of reward learning?')
    parser.add_argument('--info', type=str, default='', help='Tensorboard log is saved in ./logs/*info*_pred/true')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--test', type=bool, default=False, help='Flag to make training procedure very short (to check for errors)')
    # agent hyperparams
    parser.add_argument('--h1_agent', type=int, default=32)
    parser.add_argument('--h2_agent', type=int, default=64)
    parser.add_argument('--batch_size_agent', type=int, default=32)
    parser.add_argument('--lr_agent', type=float, default=5e-4)
    parser.add_argument('--lambda_agent', type=float, default=1e-4, help='coefficient for L2 regularization for agent optimization')
    parser.add_argument('--replay_buffer_size', type=int, default=int(1e6)) # as per Ibarz
    parser.add_argument('--target_update_period', type=int, default=8000) # as per Ibarz
    parser.add_argument('--agent_gdt_step_period', type=int, default=4) # as per Ibarz
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_start', type=float, default=0.1, help='epsilon for agent policy at start') # epsilon-annealing for agent
    parser.add_argument('--eps_final', type=float, default=0.01, help='epsilon for agent policy after eps_decay_steps')
    parser.add_argument('--eps_decay_steps', type=float, default=10**5)
    parser.add_argument('--n_initial_agent_steps', type=int, default=int(25e3), help='No. of steps that agent takes in environment during pretraining') # same as Ibarz
    parser.add_argument('--n_agent_steps', type=int, default=10**5, help='No. of steps that agent takes in environment, in main training loop')
    # parser.add_argument('--period_half_lr', type=int, default=1750) # lr is halved every period_half_lr optimizer steps

    # reward model hyperparamas
    parser.add_argument('--hid_units_rm', type=int, default=64)
    parser.add_argument('--batch_size_rm', type=int, default=16) # same as Ibarz
    parser.add_argument('--lr_rm', type=float, default=1e-4)
    parser.add_argument('--p_dropout_rm', type=float, default=0.2)
    parser.add_argument('--lambda_rm', type=float, default=1e-4, help='coefficient for L2 regularization for reward_model optimization')
    parser.add_argument('--n_epochs_pretrain_rm', type=int, default=int(50e2)) # Ibarz uses 50e3 ... but this gave me NaNs so I'm scared...
    parser.add_argument('--n_epochs_train_rm', type=int, default=6250, help='No. epochs to train reward model per round in main training loop') # as per Ibarz
    parser.add_argument('--prefs_buffer_size', type=int, default=6800) # as per Ibarz    
    parser.add_argument('--clip_length', type=int, default=25) # as per Ibarz/Christiano; i'm interested in changing this
    parser.add_argument('--use_indiff_labels', type=bool, default=True, help='Does synthetic annotator label clips about which it is indifferent as 0.5? If `False`, label equally good clips randomly')
    parser.add_argument('--corr_rollout_steps', type=int, default=1000, help='When collecting rollouts to compute correlation of true and predicted reward, how many steps per rollout?')
    parser.add_argument('--corr_num_rollouts', type=int, default=5, help='When collecting rollouts to compute correlation of true and predicted reward, how many rollouts in total?')

    # active learning
    parser.add_argument('--active_learning', type=str, default=None, help='Choice of: MC_variance, info_gain, ensemble_variance')
    parser.add_argument('--num_MC_samples', type=int, default=100)
    parser.add_argument('--size_rm_ensemble', type=int, default=1, help='If active_learning == ensemble_variance then this must be >= 2')
    parser.add_argument('--selection_factor', type=int, default=10, help='when doing active learning, 1/selection_factor of the randomly sampled clip pairs are sent to human for evaluation')
    # if doing active learning n_steps_(pre)train is automatically increased by this factor bc we consider
    # sample complexity rather than computational complexity (we assume it's cheap for the agent to do rollouts
    # and we want to find whether active learning using the same amount of *data from the human* beats the random baseline)
    args = parser.parse_args()
    if args.test:
        args.n_rounds=2
        args.n_initial_agent_steps=1000
        args.n_agent_steps=50000
        args.agent_gdt_step_period=50000
        args.n_epochs_pretrain_rm=1
        args.n_epochs_train_rm=1
    if args.active_learning == 'ensemble_variance':
        assert args.size_rm_ensemble >= 2

    return args
    

def do_pretraining(env, q_net, reward_model, prefs_buffer, args, obs_shape, act_shape, writer1, writer2):
    # Stage 0.1 Initialise policy and do some rollouts
    epsilon_pretrain = 0.1 # for now I'll use a constant epilson during pretraining
    n_initial_steps = args.selection_factor * args.n_initial_agent_steps if args.active_learning else args.n_initial_agent_steps
    num_clips = int(n_initial_steps//args.clip_length)
    assert n_initial_steps % args.clip_length == 0
    agent_experience = AgentExperience((num_clips, args.clip_length, obs_shape+act_shape), args.use_indiff_labels) # since episodes do not end we will collect one long trajectory then sample clips from it
    state = env.reset()
    for _ in tqdm(range(n_initial_steps), desc='Stage 0.1: Collecting rollouts from untrained policy, {} agent steps'.format(n_initial_steps)):
        action = q_net.act(state, epsilon_pretrain)
        assert env.action_space.contains(action)
        next_state, r_true, _, _ = env.step(action)    
        # record step information
        sa_pair = torch.tensor(np.append(state, action)).float()
        agent_experience.add(sa_pair, r_true) # add reward too in order to produce synthetic prefs
        state = next_state

    # Stage 0.2 Sample without replacement from those rollouts and label them (synthetically)
    # TODO abstract this and use the same function in training. will require some modification.
    print('Stage 0.2: Sample without replacement from those rollouts to collect {} / (2*{}) preference tuples'.format(n_initial_steps, args.clip_length))
    rand_clip_pairs, rand_rews, rand_mus = agent_experience.sample(n_initial_steps // (2*args.clip_length)) # 2 bc there are 2 clips per pair
    if args.active_learning:
        if args.active_learning == 'MC_variance':
            info_per_clip_pair = compute_MC_variance(rand_clip_pairs, reward_model, args.num_MC_samples)
        elif args.active_learning == 'info_gain':
            info_per_clip_pair = compute_entropy_reductions(rand_clip_pairs, reward_model, args.num_MC_samples)
        elif args.active_learning == 'ensemble_variance':
            info_per_clip_pair = compute_ensemble_variance(rand_clip_pairs, reward_model)
        # the following line might be stupid, as better code design would mean you don't have to recalculate num pairs
        num_pairs = len(rand_clip_pairs) // args.selection_factor # len(rand_clip_pairs) == n_initial_steps // (2*args.clip_length) == args.selection_factor * args.n_initial_steps_agent  // (2*args.clip_length) should be divisible by selection_factor anyway
        idx = np.argpartition(info_per_clip_pair, -num_pairs)[-num_pairs:] # see: tinyurl.com/ya7xr4kn
        clip_pairs, rews, mus = rand_clip_pairs[idx], rand_rews[idx], rand_mus[idx] # returned indices are not sorted
        log_active_learning(info_per_clip_pair, idx, writer1, writer2, round_num=-1)
    else:
        clip_pairs, rews, mus = rand_clip_pairs, rand_rews, rand_mus

    # put chosen clip_pairs, true rewards (just to compute mean/var of true reward across prefs_buffer)
    # and synthetic preferences into prefs_buffer
    prefs_buffer.push(clip_pairs, rews, mus)
    
    # Stage 0.3 Intialise and pretrain reward model
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm)
    reward_model.train() # dropout on
    for epoch in tqdm(range(args.n_epochs_pretrain_rm), desc='Stage 0.3: Intialise and pretrain reward model for {} batches on those preferences'.format(args.n_epochs_pretrain_rm)):
        with torch.autograd.detect_anomaly(): # detects NaNs; useful for debugging
            clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
            r_hats_batch = reward_model(clip_pair_batch).squeeze(-1)
            loss_rm = compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, obs_shape, act_shape)
            # TODO call clean version instead i.e. loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
            reward_model.train() # dropout
            optimizer_rm.zero_grad()
            loss_rm.backward()
            optimizer_rm.step()
            writer1.add_scalar('reward model loss pretraining', loss_rm, epoch)

    # evaluate reward model correlation after pretraining
    print('Reward model training complete. Evaluating reward model correlation...')
    r_xy, plots = eval_rm_correlation(reward_model, env, q_net, args, obs_shape, act_shape, rollout_steps=args.corr_rollout_steps, num_rollouts=args.corr_num_rollouts)
    log_correlation(r_xy, plots, writer1, round_num=-1)

    return reward_model, prefs_buffer


def do_training(env, q_net, q_target, reward_model, prefs_buffer, args, obs_shape, act_shape, writer1, writer2):
    # Stage 1.0: Setup
    optimizer_agent = optim.Adam(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent) # q_net initialised above
    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm) # reinitialise optimizer so we don't need to pass it between funcs
    num_clips = int(args.n_agent_steps//args.clip_length)
    assert args.n_agent_steps % args.clip_length == 0
    dummy_ep_length = env.spec.max_episode_steps

    for i_train_round in range(args.n_rounds):
        print('[Start Round {}]'.format(i_train_round))
        # Stage 1.1: Reinforcement learning with (normalised) rewards from current reward model
        q_net, replay_buffer, agent_experience = do_RL(env, q_net, q_target, optimizer_agent, replay_buffer, num_clips,
                                                        reward_model, prefs_buffer, args, i_train_round, dummy_ep_length,
                                                        obs_shape, act_shape, writer1, writer2)
        
        # Stage 1.2: Sample clip pairs without replacement from recent rollouts and label them (synthetically)
        num_labels_requested = int(58.56 * (5e6 / (i_train_round * args.n_agent_steps + 5e6) )) # compute_label_annealing_const.py
        print('Stage 1.2: Sample without replacement from those rollouts to collect {} preference tuples'.format(num_labels_requested))
        if args.active_learning:
            print('Doing Active Learning, so actually collect {} preference tuples and select the best 1/{} using {} method'.format(
                args.selection_factor * num_labels_requested, args.selection_factor, args.active_learning))
            rand_clip_pairs, rand_rews, rand_mus = agent_experience.sample(args.selection_factor * num_labels_requested)
            if args.active_learning == 'MC_variance':
                info_per_clip_pair = compute_MC_variance(rand_clip_pairs, reward_model, args.num_MC_samples)
            elif args.active_learning == 'info_gain':
                info_per_clip_pair = compute_entropy_reductions(rand_clip_pairs, reward_model, args.num_MC_samples)
            elif args.active_learning == 'ensemble_variance':
                info_per_clip_pair = compute_ensemble_variance(rand_clip_pairs, reward_model)
            idx = np.argpartition(info_per_clip_pair, -num_labels_requested)[-num_labels_requested:] # see: tinyurl.com/ya7xr4kn
            clip_pairs, rews, mus = rand_clip_pairs[idx], rand_rews[idx], rand_mus[idx] # returned indices are not sorted
            log_active_learning(info_per_clip_pair, idx, writer1, writer2, round_num=i_train_round)
        else:
            clip_pairs, rews, mus = agent_experience.sample(num_labels_requested)
        # put labelled clip_pairs into prefs_buffer
        assert len(clip_pairs) == num_labels_requested
        prefs_buffer.push(clip_pairs, rews, mus)
        
        # Stage 1.3: Train reward model
        reward_model.train() # dropout on
        for epoch in tqdm(range(args.n_epochs_train_rm), desc='Stage 1.3: Train reward model for {} batches on those preferences'.format(args.n_epochs_train_rm)):
            with torch.autograd.detect_anomaly():
                clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
                r_hats_batch = reward_model(clip_pair_batch).squeeze(-1) # squeeze the oa_pair dimension that was passed through reward_model
                assert clip_pair_batch.shape == (args.batch_size_rm, 2, args.clip_length, obs_shape + act_shape)
                loss_rm = compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, obs_shape, act_shape)
                # TODO call clean version instead i.e. loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
                optimizer_rm.zero_grad()
                loss_rm.backward()
                optimizer_rm.step()
                writer1.add_scalar('reward model loss/round {}'.format(i_train_round), loss_rm, epoch)

        # evaluate reward model correlation
        print('Reward model training complete. Evaluating reward model correlation...')
        r_xy, plots = eval_rm_correlation(reward_model, env, q_net, args, obs_shape, act_shape, rollout_steps=args.corr_rollout_steps, num_rollouts=args.corr_num_rollouts)
        log_correlation(r_xy, plots, writer1, round_num=i_train_round)


def main(): 
    # experiment settings
    args = parse_arguments()

    # for reproducibility
    torch.manual_seed(args.random_seed) # TODO check that setting random seed here also applies to random calls in modules
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # TensorBoard logging
    logdir = './logs/test/' if args.test else './logs/'
    writer1 = SummaryWriter(log_dir=logdir+'{}_pred'.format(args.info))
    writer2 = SummaryWriter(log_dir=logdir+'{}_true'.format(args.info))

    # make environment
    env = gym.make(args.env_class, ep_end_penalty=-10.0)
    obs_shape = env.observation_space.shape[0] # env.observation_space is Box(4,) and calling .shape returns (4,) [gym can be ugly]
    assert isinstance(env.action_space, gym.spaces.Discrete), 'DQN requires discrete action space.'
    act_shape = 1 # [gym doesn't have a nice way to get shape of Discrete space... env.action_space.shape -> () ]
    n_actions = env.action_space.n # env.action_space is Discrete(2) and calling .n returns 2

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

    # fire away!
    reward_model, prefs_buffer = do_pretraining(env, q_net, reward_model, prefs_buffer, args, obs_shape, act_shape, writer1, writer2)
    do_training(env, q_net, q_target, reward_model, prefs_buffer, args, obs_shape, act_shape, writer1, writer2)
    
    writer1.close()
    writer2.close()

if __name__ == '__main__':
    main()