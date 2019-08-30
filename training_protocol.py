"""Components of Deep RL from Human Preferences training protocol"""

import math, logging
from collections import Counter
from time import time
import pandas as pd
from gym import wrappers
from q_learning import *
from reward_learning import *
from reinforcement_learning import *
from reward_learning_utils import *
from active_learning import *
from test_policy import *
from annotation import *
from rl_logging import *
from utils import *

def training_protocol(env, args, writers, returns_summary, i_run):
    """Implements Algorithm 1 in Ibarz et al. (2018)
       with the modification that labels on clip pairs
       may be acquired sequentially i.e. reward model is retrained
       after each new label is acquired, then that retrained
       model is used to select the next clip pair for labelling.
    """
    # SET UP: instantiate reward model + buffers and optimizers for training DQN and reward model
    reward_model, optimizer_rm = init_rm(args)
    prefs_buffer = PrefsBuffer(args)
    q_net, q_target, replay_buffer, optimizer_agent, epsilon_schedule = init_agent(args)
    true_reward_stats = TrueRewardRunningStat()
    
    # BEGIN PRETRAINING
    # Stage 0.1: Do some rollouts from the randomly initialised policy
    agent_experience, replay_buffer = do_pretraining_rollouts(q_net, replay_buffer, env, args)
    # Stage 0.2: Sample without replacement from those rollouts and label them (synthetically)
    mu_counts_total = np.zeros((2,3))
    if not args.RL_baseline or args.normalise_rewards:
        reward_model, prefs_buffer, mu_counts_total, true_reward_stats = acquire_labels_and_train_rm(
            agent_experience, reward_model, prefs_buffer, optimizer_rm,
            args, writers, mu_counts_total, true_reward_stats, i_train_round=0)
    
    # evaluate reward model correlation after pretraining (currently not interested)
    # if not args.RL_baseline:
    #     test_correlation(reward_model, env, q_net, args, writers[0], i_train_round=-1)

    # BEGIN TRAINING
    for i_train_round in range(1, args.n_rounds+1):
        logging.info('[Start Round {}]'.format(i_train_round))
        # Stage 1.1a: Reinforcement learning with (normalised) rewards from current reward model
        log_agent_training_info(args, i_train_round)
        if args.reinit_agent:
            q_net, q_target, _, optimizer_agent, epsilon_schedule = init_agent(args) # keep replay buffer experience -- OK as long as we use the new rewards
        # set up buffer to collect agent_experience for possible annotation
        num_clips = args.n_agent_steps // args.clip_length
        agent_experience = AgentExperience(num_clips, args) # since episodes do not end we collect one long trajectory then sample clips from it. TODO I can now init this *inside* do_RL since I refactored things nicely
        
        q_net, q_target, replay_buffer, agent_experience = do_RL(env, q_net, q_target, optimizer_agent, replay_buffer, epsilon_schedule,
                                                                 agent_experience, reward_model, returns_summary,
                                                                 true_reward_stats, args, writers, i_run, i_train_round)

        # Stage 1.2 - 1.3: acquire labels from recent rollouts and train reward model on current dataset
        if i_train_round != args.n_rounds and (not args.RL_baseline or args.normalise_rewards):
            reward_model, prefs_buffer, mu_counts_total, true_reward_stats = acquire_labels_and_train_rm(
                agent_experience, reward_model, prefs_buffer, optimizer_rm,
                args, writers, mu_counts_total, true_reward_stats, i_train_round)
        
        pd.DataFrame(returns_summary).to_csv('./logs/{}.csv'.format(args.info), index_label=['ep return type', 'round no.', 'test no.'])
        # Evaluate reward model correlation (currently not interested)
        # if not args.RL_baseline:
        #     test_correlation(reward_model, env, q_net, args, writers[0], i_train_round)

    # log mu_counts for this experiment
    if not args.RL_baseline: log_total_mu_counts(mu_counts_total, writers, args)


def acquire_labels_and_train_rm(agent_experience, reward_model, prefs_buffer, optimizer_rm, args, writers, mu_counts_total, true_reward_stats, i_train_round):
    n_labels = args.n_labels_per_round[i_train_round]
    batch_size_acq = args.batch_size_acq[i_train_round]
    n_acq_batches = args.n_acq_batches_per_round[i_train_round]
    n_labels_so_far = sum(args.n_labels_per_round[:i_train_round])
    logging.info('Stage {}.2: Sample without replacement from those rollouts to collect {} labels/preference tuples'.format(i_train_round, n_labels))
    logging.info('The reward model will be retrained after every batch of label acquisitions')
    logging.info('Making {} acquisitions, consisting of {} batches of acquisitions of size {}'.format(
        n_labels, n_acq_batches, batch_size_acq))
    rand_clip_data = generate_rand_clip_pairing(agent_experience, n_labels, args)
    logging.info('Stage {}.3: Training reward model for {} sets of batches on those preferences'.format(i_train_round, n_acq_batches))
    for i_acq_batch in range(n_acq_batches):
        i_acq = n_labels_so_far + i_acq_batch
        acquired_clip_data, idx, mu_counts_total = make_acquisitions(
            rand_clip_data, batch_size_acq, reward_model, args, writers, mu_counts_total, i_acq)
        prefs_buffer.push(*acquired_clip_data)
        rand_clip_data = remove_acquisitions(idx, rand_clip_data)
        if args.reinit_rm:
            reward_model, optimizer_rm = init_rm(args)
        # Train reward model!
        reward_model = train_reward_model(reward_model, prefs_buffer, optimizer_rm, args, writers, i_acq)
        # update running mean and variance of reward model based on these acquisitions
        logging.info('Update running mean and variance with {} acquired clip pairs'.format(
            len(acquired_clip_data[0])))
        reward_model = update_running_mean_var(reward_model, acquired_clip_data)
        true_reward_stats.push_clip_pairs(acquired_clip_data)
    # save reward_model for loading later
    save_reward_model(reward_model, optimizer_rm, i_train_round, args)
    save_preferece_info(prefs_buffer, true_reward_stats, i_train_round, args)
    return reward_model, prefs_buffer, mu_counts_total, true_reward_stats


def do_pretraining_rollouts(q_net, replay_buffer, env, args):
    """Agent interact with environment and collect experience.
       Number of steps taken determined by `n_labels_pretraining`.
       Currently used only in pretraining, but I might refactor s.t. there's
       a single function that I can use for agent-environment
       interaction (with or without training).
    """
    n_labels_pretraining = args.n_labels_per_round[0]
    n_steps_to_collect_enough_clips = args.selection_factor * n_labels_pretraining * 2 * args.clip_length
    n_initial_steps = max(args.n_agent_steps_pretrain, n_steps_to_collect_enough_clips)
    assert n_initial_steps % args.clip_length == 0,\
        "You should specify a number of initial steps ({}) that is divisible by args.clip_length ({})".format(
            n_initial_steps, args.clip_length)
    num_clips = n_initial_steps // args.clip_length
    logging.info('Stage -1.1: Collecting rollouts from untrained policy, {} agent steps'.format(n_initial_steps))
    agent_experience = AgentExperience(num_clips, args)
    epsilon_pretrain = 1 # for now I'll use a constant epilson during pretraining
    state = env.reset()
    for _ in range(n_initial_steps):
        action = q_net.act(state, epsilon_pretrain)
        assert env.action_space.contains(action)
        next_state, r_true, done, _ = env.step(action)    
        # record step information
        # sa_pair = torch.tensor(np.append(state, action)).float() # old code. float32s used too much memory when observation shape too large
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_one_hot = one_hot_action(action, env)
        sa_pair = np.append(state, action_one_hot).astype(args.oa_dtype, casting='unsafe') # casting='unsafe' is default; i want to make explicit that this could be a dangerous line
        assert (sa_pair == np.append(state, action_one_hot)).all() # check casting done safely. should be redundant since i set oa_dtype based on env, earlier. but you can never be too careful since this would fail silently!
        agent_experience.add(sa_pair, r_true) # add reward too in order to produce synthetic prefs
        if args.agent_gets_dones:
            replay_buffer.push(state, action, r_true, next_state, done)
        else:
            replay_buffer.push(state, action, r_true, next_state, False) # done=False since agent thinks the task is continual; r_true used only when args.RL baseline
        state = next_state
        if done:
            state = env.reset()
    return agent_experience, replay_buffer


def do_random_experiment(env, args, returns_summary, writers, i_run):
    writer1, _ = writers
    for i_train_round in range(args.n_rounds):
        logging.info('[Start Round {}]'.format(i_train_round))
        returns = {'ep': 0, 'all': []}
        logging.info('Taking random actions for {} steps'.format(args.n_agent_steps))
        env.reset()
        i_test = 0
        for step in range(args.n_agent_steps):
            action = env.action_space.sample()
            _, r_true, done, _ = env.step(action)
            returns['ep'] += r_true
            if done:
                state = env.reset()
                writer1.add_scalar('4a.train_ep_return_per_step/round_{}'.format(i_train_round), returns['ep'], step)
                returns['all'].append(returns['ep'])
                returns['ep'] = 0
            if step % args.agent_test_period == 0:
                # Eval random agent performance
                logging.info("Agent has taken {} steps. Testing performance for 100 episodes".format(step))
                test_and_log_random_policy(returns_summary, step, i_test, i_train_round, i_run, writers, args)
                i_test += 1