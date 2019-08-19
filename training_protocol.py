"""Components of Deep RL from Human Preferences training protocol"""

import math, logging
from collections import Counter
from time import time
import pandas as pd
from gym import wrappers
from q_learning import *
from reward_learning import *
from active_learning import *
from test_policy import *
from annotation import *
from rl_logging import *

def training_protocol(env, args, writers, returns_summary, i_run):
    """Implements Algorithm 1 in Ibarz et al. (2018)
       with the modification that labels on clip pairs
       may be acquired sequentially i.e. reward model is retrained
       after each new label is acquired, then that retrained
       model is used to select the next clip pair for labelling.
    """
    # SET UP: instantiate reward model + buffers and optimizers for training DQN and reward model
    reward_model, optimizer_rm = init_rm(args)
    # prefs_buffer = PrefsBuffer(capacity=args.prefs_buffer_size, clip_shape=(args.clip_length, args.obs_act_shape))
    prefs_buffer = PrefsBuffer(args)
    q_net, q_target, replay_buffer, optimizer_agent = init_agent(args)
    
    # BEGIN PRETRAINING
    # Stage 0.1: Do some rollouts from the randomly initialised policy
    agent_experience, replay_buffer = do_pretraining_rollouts(q_net, replay_buffer, env, args)
    # Stage 0.2: Sample without replacement from those rollouts and label them (synthetically)
    mu_counts_total = np.zeros((2,3))
    reward_model, reward_stats, prefs_buffer, mu_counts_total = acquire_labels_and_train_rm(
        agent_experience, reward_model, prefs_buffer, optimizer_rm, args, writers, mu_counts_total, i_train_round=-1)

    # evaluate reward model correlation after pretraining (currently not interested)
    # if not args.RL_baseline:
    #     test_correlation(reward_model, env, q_net, args, writers[0], i_train_round=-1)

    # BEGIN TRAINING
    for i_train_round in range(args.n_rounds):
        logging.info('[Start Round {}]'.format(i_train_round))
        # Stage 1.1a: Reinforcement learning with (normalised) rewards from current reward model
        log_agent_training_info(args, i_train_round)
        if args.reinit_agent:
            q_net, q_target, _, optimizer_agent = init_agent(args) # keep replay buffer experience -- OK as long as we use the new rewards
        # set up buffer to collect agent_experience for possible annotation
        num_clips = args.n_agent_steps // args.clip_length
        agent_experience = AgentExperience(num_clips, args) # since episodes do not end we collect one long trajectory then sample clips from it
        for sub_round in range(args.agent_test_frequency): # code more readable if this for-loop converted to if-statement
            logging.info("Begin train {}".format(sub_round))
            q_net, q_target, replay_buffer, agent_experience = do_RL(env, q_net, q_target, optimizer_agent, replay_buffer,
                                                                     agent_experience, reward_model, reward_stats, args,
                                                                     writers, i_train_round, sub_round)
            # Stage 1.1b: Evalulate RL agent performance
            logging.info("Begin test {}".format(sub_round))
            test_returns = test_policy(q_net, reward_model, reward_stats, args, writers, i_train_round, sub_round)
            mean_ret_true = log_tested_policy(test_returns, writers, returns_summary, args, i_run, i_train_round, sub_round, env)
            # Possibly end training if mean_ret_true is above the threshold
            if not args.continue_once_solved and env.spec.reward_threshold != None:
                if mean_ret_true >= env.spec.reward_threshold:
                    raise SystemExit("Environment solved, moving onto next run.")

        # Stage 1.2 - 1.3: acquire labels from recent rollouts and train reward model on current dataset
        reward_model, reward_stats, prefs_buffer, mu_counts_total = acquire_labels_and_train_rm(
            agent_experience, reward_model, prefs_buffer, optimizer_rm, args, writers, mu_counts_total, i_train_round)
        
        pd.DataFrame(returns_summary).to_csv('./logs/{}.csv'.format(args.info), index_label=['ep return type', 'round no.', 'test no.'])
            # Evaluate reward model correlation (currently not interested)
        # if not args.RL_baseline:
        #     test_correlation(reward_model, env, q_net, args, writers[0], i_train_round)

    # log mu_counts for this run
    log_total_mu_counts(mu_counts_total, writers, args)


def acquire_labels_and_train_rm(agent_experience, reward_model, prefs_buffer, optimizer_rm, args, writers, mu_counts_total, i_train_round):
    logging.info('Stage {}.2: Sample without replacement from those rollouts to collect {} labels/preference tuples'.format(i_train_round, args.n_labels_per_round))
    logging.info('The reward model will be retrained after every batch of label acquisitions')
    logging.info('Making {} acquisitions, consisting of {} batches of acquisitions of size {}'.format(
        args.n_labels_per_round, args.n_acq_batches_per_round, args.batch_size_acq))
    rand_clip_data = generate_rand_clip_pairing(agent_experience, args.n_labels_per_round, args)
    logging.info('Stage {}.3: Training reward model for {} sets of batches on those preferences'.format(i_train_round, args.n_acq_batches_per_round))
    for i_acq_batch in range(args.n_acq_batches_per_round):
        i_label = args.n_acq_batches_per_round * i_train_round + i_acq_batch
        prefs_buffer, rand_clip_data, mu_counts_total = make_acquisitions(rand_clip_data, reward_model, prefs_buffer, args, writers, mu_counts_total, i_label)
        if args.reinit_rm:
            reward_model, optimizer_rm = init_rm(args)
        # Train reward model!
        # if prefs_buffer.current_length >= 10: # prevent gradient updates if too few training examples
        reward_model = train_reward_model(reward_model, prefs_buffer, optimizer_rm, args, writers, i_label)
    # Compute mean and variance of true and predicted reward after training (for normalising rewards sent to agent)
    reward_stats, reward_model = compute_reward_stats(reward_model, prefs_buffer)
    return reward_model, reward_stats, prefs_buffer, mu_counts_total


def do_RL(env, q_net, q_target, optimizer_agent, replay_buffer,
          agent_experience, reward_model, reward_stats, args,
          writers, i_train_round, sub_round):
    writer1, writer2 = writers
    dummy_returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
                    'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}
    # Do RL!
    state = env.reset()
    is_saving_video = False
    done_saving_video = False
    start_step = sub_round * args.n_agent_steps_before_test
    step_range = range(start_step, start_step + args.n_agent_steps_before_test)
    for step in step_range:
        # agent interact with env
        epsilon = args.exploration.value(step)
        action = q_net.act(state, epsilon)
        assert env.action_space.contains(action)
        next_state, r_true, done, _ = env.step(action) # one continuing episode
        # record step info
        # sa_pair = torch.tensor(np.append(state, action)).float()
        sa_pair = np.append(state, action).astype(args.oa_dtype, casting='unsafe')
        assert (sa_pair == np.append(state, action)).all() # check casting done safely. should be redundant since i set oa_dtype based on env, earlier. but you can never be too careful since this would fail silently!
        agent_experience.add(sa_pair, r_true) # include reward in order to later produce synthetic prefs
        replay_buffer.push(state, action, r_true, next_state, False) # done=False since agent thinks the task is continual; r_true used only when args.RL baseline
        dummy_returns = log_agent_step(sa_pair, r_true, dummy_returns, reward_stats, reward_model, args)
        # prepare for next step
        state = next_state
        if done:
            if args.save_video and not (is_saving_video or done_saving_video) and (args.n_agent_steps_before_test - step < 4*env.spec.max_episode_steps):
                # save the final 4ish train episodes (see https://github.com/openai/gym/wiki/FAQ#how-do-i-export-the-run-to-a-video-file)
                env = wrappers.Monitor(env, args.logdir + '/videos/train/' + str(time()) + '/')
                is_saving_video = True # don't call the env wrapper again
            # if is_saving_video and step >= args.n_agent_train_steps_before_test: # don't need this anymore as i take non-training steps at start
            #     env = gym.make(args.env_ID) # unwrap Monitor wrapper during non-training steps
            #     env.seed(args.random_seed)
            #     done_saving_video = True
            state = env.reset()
            dummy_returns = log_agent_episode(dummy_returns, writers, step, i_train_round, sub_round, args, is_test=False)

        # q_net gradient step
        if step >= args.agent_learning_starts and step % args.agent_gdt_step_period == 0 and \
                len(replay_buffer) >= 3*args.batch_size_agent:
            if args.RL_baseline:
                loss_agent = q_learning_loss(q_net, q_target, replay_buffer, args, normalise_rewards=args.normalise_rewards, true_reward_stats=reward_stats)
            else:
                loss_agent = q_learning_loss(q_net, q_target, replay_buffer, args, reward_model=reward_model, normalise_rewards=args.normalise_rewards)
            optimizer_agent.zero_grad()
            loss_agent.backward()
            optimizer_agent.step()
            writer1.add_scalar('7.agent_loss/round_{}'.format(i_train_round), loss_agent, step)
            # scheduler.step() # Ibarz doesn't mention lr annealing...
            writer1.add_scalar('8.agent_epsilon/round_{}'.format(i_train_round), epsilon, step)
            # if q_net.epsilon > q_net.epsilon_stop:
            #     q_net.epsilon *= q_net.epsilon_decay

        # update q_target
        if step % args.target_update_period == 0: # update target parameters
            for target_param, local_param in zip(q_target.parameters(), q_net.parameters()):
                target_param.data.copy_(q_net.tau*local_param.data + (1.0-q_net.tau)*target_param.data)
            # q_target.load_state_dict(q_net.state_dict()) # old hard update code            

    # log mean return this training round
    log_RL_loop(dummy_returns, args, i_train_round, sub_round, writers)
    
    return q_net, q_target, replay_buffer, agent_experience


def do_pretraining_rollouts(q_net, replay_buffer, env, args):
    """Agent interact with environment and collect experience.
       Number of steps taken determined by `args.n_labels_per_round`.
       NB Used to be determined by `args.n_labels_pretraining` until
       I dropped support for that.
       Currently used only in pretraining, but I might refactor s.t. there's
       a single function that I can use for agent-environment
       interaction (with or without training).
    """
    n_steps_to_collect_enough_clips = args.selection_factor * args.n_labels_per_round * 2 * args.clip_length
    n_initial_steps = max(args.n_agent_steps_pretrain, n_steps_to_collect_enough_clips)
    assert n_initial_steps % args.clip_length == 0,\
        "You should specify a number of initial steps ({}) that is divisible by args.clip_length ({})".format(
            n_initial_steps, args.clip_length)
    num_clips = n_initial_steps // args.clip_length
    logging.info('Stage -1.1: Collecting rollouts from untrained policy, {} agent steps'.format(n_initial_steps))
    agent_experience = AgentExperience(num_clips, args)
    epsilon_pretrain = 0.5 # for now I'll use a constant epilson during pretraining
    state = env.reset()
    for _ in range(n_initial_steps):
        action = q_net.act(state, epsilon_pretrain)
        assert env.action_space.contains(action)
        next_state, r_true, done, _ = env.step(action)    
        # record step information
        # sa_pair = torch.tensor(np.append(state, action)).float() # old code. float32s used too much memory when observation shape too large
        sa_pair = np.append(state, action).astype(args.oa_dtype, casting='unsafe') # casting='unsafe' is default; i want to make explicit that this could be a dangerous line
        assert (sa_pair == np.append(state, action)).all() # check casting done safely. should be redundant since i set oa_dtype based on env, earlier. but you can never be too careful since this would fail silently!
        agent_experience.add(sa_pair, r_true) # add reward too in order to produce synthetic prefs
        replay_buffer.push(state, action, r_true, next_state, False) # done=False since agent thinks the task is continual; r_true used only when args.RL baseline
        state = next_state
        if done:
            state = env.reset()
    return agent_experience, replay_buffer


def do_random_experiment(env, args, returns_summary, writers, i_run):
    """TODO refactor this s.t. I can reuse training_protocol()?
       At least make training_protocol() more friendly to slotting
       in different agents. I'll need to do this when I want to use
       SAC, anyway.
    """
    writer1, _ = writers
    for i_train_round in range(args.n_rounds):
        logging.info('[Start Round {}]'.format(i_train_round))
        dummy_returns = {'ep': 0, 'all': []}
        env.reset()
        logging.info('Taking random actions for {} steps'.format(args.n_agent_steps))
        for step in range(args.n_agent_steps):
            # agent interact with env
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            _, r_true, done, _ = env.step(action) # one continuous episode
            dummy_returns['ep'] += r_true # record step info

            if done:
                state = env.reset()
                writer1.add_scalar('4a.train_ep_return_per_step/round_{}'.format(i_train_round), dummy_returns['ep'], step)
                dummy_returns['all'].append(dummy_returns['ep'])
                dummy_returns['ep'] = 0

        # log mean recent return this training round
        # mean_dummy_true_returns = np.sum(np.array(dummy_returns['all'][-3:])) / 3. # 3 dummy eps is the final 3*200/2000 == 3/10 eps in the round
        mean_dummy_true_returns = np.sum(np.array(dummy_returns['all'])) / len(dummy_returns['all'])
        writer1.add_scalar('4a.train_mean_ep_return_per_round', mean_dummy_true_returns, i_train_round)
        test_and_log_random_policy(writers, returns_summary, args, i_run, i_train_round)