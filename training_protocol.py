"""Components of Deep RL from Human Preferences training protocol"""

import math, logging
from collections import Counter
from q_learning import *
from reward_learning import *
from active_learning import *
from test_policy import *
from annotation import *

def training_protocol(env, args, writers, returns_summary, i_run):
    """Implements Algorithm 1 in Ibarz et al. (2018)
       with the modification that labels on clip pairs
       may be acquired sequentially i.e. reward model is retrained
       after each new label is acquired, then that retrained
       model is used to select the next clip pair for labelling.
    """
    # SET UP: instantiate reward model + buffers and optimizers for training DQN and reward model
    reward_model, optimizer_rm = init_rm(args)
    prefs_buffer = PrefsBuffer(capacity=args.prefs_buffer_size, clip_shape=(args.clip_length, args.obs_act_shape))
    q_net, q_target, replay_buffer, optimizer_agent = init_agent(args)
    
    # BEGIN PRETRAINING
    # Stage 0.1: Do some rollouts from the randomly initialised policy
    agent_experience = do_pretraining_rollouts(q_net, env, args)
    # Stage 0.2: Sample without replacement from those rollouts and label them (synthetically)
    mu_counts_total = np.zeros((2,3))
    reward_model, prefs_buffer, mu_counts_total = acquire_labels_and_train_rm(
            agent_experience, reward_model, prefs_buffer, optimizer_rm, args, writers, mu_counts_total, i_train_round=-1)

    # evaluate reward model correlation after pretraining (currently not interested)
    # if not args.RL_baseline:
    #     test_correlation(reward_model, env, q_net, args, writers[0], i_train_round=-1)

    # BEGIN TRAINING
    for i_train_round in range(args.n_rounds):
        logging.info('[Start Round {}]'.format(i_train_round))
        # Compute mean and variance of true and predicted reward (for normalising rewards sent to agent)
        reward_stats, reward_model = compute_reward_stats(reward_model, prefs_buffer)
        # TODO when prefs_buffer is small, reward_stats may be weird and hinder performance..?
        # (in Ibarz, it always has at least 50 or 100 examples...)
        # Stage 1.1a: Reinforcement learning with (normalised) rewards from current reward model
        if args.reinit_agent:
            q_net, q_target, _, optimizer_agent = init_agent(args) # keep replay buffer experience -- OK as long as we use the new rewards
        q_net, q_target, replay_buffer, agent_experience = do_RL(env, q_net, q_target, optimizer_agent, replay_buffer,
                                                                 reward_model, prefs_buffer, reward_stats, args, writers, i_train_round)
        # Stage 1.1b: Evalulate RL agent performance
        test_returns = test_policy(q_net, reward_model, reward_stats, args)
        mean_ret_true = log_tested_policy(test_returns, writers, returns_summary, args, i_run, i_train_round)
        
        # Possibly end training if mean_ret_true is high enough
        if args.terminate_once_solved:
            if mean_ret_true >= env.spec.reward_threshold:
                raise SystemExit("Environment solved, moving onto next run.")

        # Stage 1.2 - 1.3: acquire labels from recent rollouts and train reward model on current dataset
        reward_model, prefs_buffer, mu_counts_total = acquire_labels_and_train_rm(
            agent_experience, reward_model, prefs_buffer, optimizer_rm, args, writers, mu_counts_total, i_train_round)
        
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
    logging.info('Stage {}.3: Training reward model for {}*{} batches on those preferences'.format(i_train_round, args.n_acq_batches_per_round, args.n_epochs_train_rm))
    for i_acq_batch in range(args.n_acq_batches_per_round):
        i_label = args.n_acq_batches_per_round * i_train_round + i_acq_batch
        prefs_buffer, rand_clip_data, mu_counts_total = make_acquisitions(rand_clip_data, reward_model, prefs_buffer, args, writers, mu_counts_total, i_label)
        if args.reinit_rm:
            reward_model, optimizer_rm = init_rm(args)
        reward_model = train_reward_model(reward_model, prefs_buffer, optimizer_rm, args, writers, i_label)
    return reward_model, prefs_buffer, mu_counts_total


def do_RL(env, q_net, q_target, optimizer_agent, replay_buffer, reward_model, prefs_buffer, reward_stats, args, writers, i_train_round):
    writer1, writer2 = writers
    rt_mean, rt_var = reward_stats
    # log some info
    if args.RL_baseline:
        logging.info('Stage {}.1: RL using *true reward*'.format(i_train_round))
    else:
        logging.info('Stage {}.1: RL using reward model'.format(i_train_round))
    logging.info('Training will last {} steps, of which in the first {} we make a learning update every {} step(s)'.format(
                args.n_agent_total_steps, args.n_agent_train_steps, args.agent_gdt_step_period))

    # set up buffer to collect agent_experience for possible annotation
    num_clips = int(args.n_agent_total_steps // args.clip_length)
    assert args.n_agent_total_steps % args.clip_length == 0,\
    "The agent should take a number of steps that is divisible by the clip length. Currently, agent takes {} steps but clip length = {}".format(
        args.n_agent_total_steps, args.clip_length)
    agent_experience = AgentExperience((num_clips, args.clip_length, args.obs_act_shape), args.force_label_choice) # since episodes do not end we will collect one long trajectory then sample clips from it     
    dummy_returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
                    'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}
    # Do RL!
    state = env.reset()
    for step in range(args.n_agent_total_steps):
        # agent interact with env
        action = q_net.act(state, q_net.epsilon)
        assert env.action_space.contains(action)
        next_state, r_true, _, _ = env.step(action) # one continuing episode
        # record step info
        sa_pair = torch.tensor(np.append(state, action)).float()
        agent_experience.add(sa_pair, r_true) # include reward in order to later produce synthetic prefs
        if step < args.n_agent_train_steps:
            replay_buffer.push(state, action, r_true, next_state, False) # reward used to check against RL baseline; done=False since agent is in one continuous episode
            dummy_returns['ep']['true'] += r_true
            dummy_returns['ep']['true_norm'] += (r_true - rt_mean) / np.sqrt(rt_var + 1e-8) # TODO make a custom class for this, np array with size fixed in advance, given 2 means and vars, have it do the normalisation automatically and per batch (before logging)
            # also log reward the agent thinks it's getting according to current reward_model
            if not args.RL_baseline:
                reward_model.eval() # dropout off at 'test' time i.e. when logging performance
                r_pred = reward_model(sa_pair).item()
                r_pred_norm = reward_model(sa_pair, normalise=True).item()
                dummy_returns['ep']['pred'] += r_pred
                dummy_returns['ep']['pred_norm'] += r_pred_norm
        # prepare for next step
        state = next_state

        # log performance after a "dummy" episode has elapsed
        if (step + 1) % args.dummy_ep_length == 0 and step < args.n_agent_train_steps:
            # interpreting writers: 1 == blue == true!
            writer1.add_scalar('3a.dummy_ep_return_against_step/round_{}'.format(i_train_round), dummy_returns['ep']['true'], step+1)
            writer1.add_scalar('3b.dummy_ep_return_against_step_normalised/round_{}'.format(i_train_round), dummy_returns['ep']['true_norm'], step+1)
            if not args.RL_baseline:
                writer2.add_scalar('3a.dummy_ep_return_against_step/round_{}'.format(i_train_round), dummy_returns['ep']['pred'], step+1)
                writer2.add_scalar('3b.dummy_ep_return_against_step_normalised/round_{}'.format(i_train_round), dummy_returns['ep']['pred_norm'], step+1)
            for key, value in dummy_returns['ep'].items():
                dummy_returns['all'][key].append(value)
                dummy_returns['ep'][key] = 0

        # q_net gradient step
        if step % args.agent_gdt_step_period == 0 and len(replay_buffer) >= 3*q_net.batch_size and step < args.n_agent_train_steps:
            if args.RL_baseline:
                loss_agent = q_learning_loss(q_net, q_target, replay_buffer, args, normalise_rewards=True, rt_mean=rt_mean, rt_var=rt_var)
            else:
                loss_agent = q_learning_loss(q_net, q_target, replay_buffer, args, reward_model=reward_model, normalise_rewards=True)
            optimizer_agent.zero_grad()
            loss_agent.backward()
            optimizer_agent.step()
            # decay epsilon every learning step
            writer1.add_scalar('9.agent_epsilon/round_{}'.format(i_train_round), q_net.epsilon, step)
            if q_net.epsilon > q_net.epsilon_stop:
                q_net.epsilon *= q_net.epsilon_decay
            writer1.add_scalar('8.agent_loss/round_{}'.format(i_train_round), loss_agent, step)
            # scheduler.step() # Ibarz doesn't mention lr annealing...

        # update q_target
        if step % args.target_update_period == 0 and step < args.n_agent_train_steps: # soft update target parameters
            for target_param, local_param in zip(q_target.parameters(), q_net.parameters()):
                target_param.data.copy_(q_net.tau*local_param.data + (1.0-q_net.tau)*target_param.data)
            # q_target.load_state_dict(q_net.state_dict()) # old hard update code

    # log mean recent return this training round
    # mean_dummy_true_returns = np.sum(np.array(dummy_returns['all']['true'][-3:])) / 3. # 3 dummy eps is the final 3*200/2000 == 3/10 eps in the round
    mean_dummy_true_returns = np.sum(np.array(dummy_returns['all']['true'])) / len(dummy_returns['all']['true'])
    writer1.add_scalar('2.dummy_mean_ep_returns_per_training_round', mean_dummy_true_returns, i_train_round)
    if not args.RL_baseline:
        # mean_dummy_pred_returns = np.sum(np.array(dummy_returns['all']['pred'][-3:])) / 3.
        mean_dummy_pred_returns = np.sum(np.array(dummy_returns['all']['pred'])) / len(dummy_returns['all']['pred'])
        writer2.add_scalar('2.dummy_mean_ep_returns_per_training_round', mean_dummy_pred_returns, i_train_round)
    
    return q_net, q_target, replay_buffer, agent_experience


def do_pretraining_rollouts(q_net, env, args):
    """Agent interact with environment and collect experience.
       Number of steps taken determined by `args.n_labels_per_round`.
       NB Used to be determined by `args.n_labels_pretraining` until
       I dropped support for that.
       Currently used only in pretraining, but I might refactor s.t. there's
       a single function that I can use for agent-environment
       interaction (with or without training).
    """
    n_initial_steps = args.selection_factor * args.n_labels_per_round * 2 * args.clip_length
    num_clips       = args.selection_factor * args.n_labels_per_round * 2
    logging.info('Stage -1.1: Collecting rollouts from untrained policy, {} agent steps'.format(n_initial_steps))
    agent_experience = AgentExperience((num_clips, args.clip_length, args.obs_act_shape), args.force_label_choice)
    epsilon_pretrain = 0.5 # for now I'll use a constant epilson during pretraining
    state = env.reset()
    for _ in range(n_initial_steps):
        action = q_net.act(state, epsilon_pretrain)
        assert env.action_space.contains(action)
        next_state, r_true, _, _ = env.step(action)    
        # record step information
        sa_pair = torch.tensor(np.append(state, action)).float()
        agent_experience.add(sa_pair, r_true) # add reward too in order to produce synthetic prefs
        state = next_state
    return agent_experience


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
        logging.info('Taking random actions for {} steps'.format(args.n_agent_train_steps))
        for step in range(args.n_agent_train_steps):
            # agent interact with env
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            _, r_true, _, _ = env.step(action) # one continuous episode
            dummy_returns['ep'] += r_true # record step info

            # log performance after a "dummy" episode has elapsed
            if (step % args.dummy_ep_length == 0 or step == args.n_agent_train_steps - 1):
                writer1.add_scalar('3a.dummy_ep_return_against_step/round_{}'.format(i_train_round), dummy_returns['ep'], step)
                dummy_returns['all'].append(dummy_returns['ep'])
                dummy_returns['ep'] = 0

        # log mean recent return this training round
        # mean_dummy_true_returns = np.sum(np.array(dummy_returns['all'][-3:])) / 3. # 3 dummy eps is the final 3*200/2000 == 3/10 eps in the round
        mean_dummy_true_returns = np.sum(np.array(dummy_returns['all'])) / len(dummy_returns['all'])
        writer1.add_scalar('2.dummy_mean_ep_returns_per_training_round', mean_dummy_true_returns, i_train_round)
        test_and_log_random_policy(writers, returns_summary, args, i_run, i_train_round)