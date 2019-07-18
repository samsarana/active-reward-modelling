"""Components of Deep RL from Human Preferences training protocol"""

import logging
from q_learning import *
from reward_learning import *
from active_learning import *
from test_policy import *

def training_protocol(env, q_net, q_target, args, writers, returns_summary, i_run):
    """Implements Algorithm 1 in Ibarz et al. (2018)
    """
    # SET UP: instantiate reward model + buffers and optimizers for training DQN and reward model
    if args.size_rm_ensemble >= 2:
        reward_model = RewardModelEnsemble(args.obs_shape, args.act_shape, args)
        logging.info('Using a {}-ensemble of nets for our reward model'.format(args.size_rm_ensemble))
    else:
        reward_model = RewardModel(args.obs_shape, args.act_shape, args)
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm)
    prefs_buffer = PrefsBuffer(capacity=args.prefs_buffer_size, clip_shape=(args.clip_length, args.obs_act_shape))

    optimizer_agent = optim.Adam(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent)
    replay_buffer = ReplayBuffer(args.replay_buffer_size) # TODO change this (and several other things) s.t. we can use different RL agents
    
    # BEGIN PRETRAINING
    # Stage 0.1 Do some rollouts from the randomly initialised policy
    agent_experience = do_pretraining_rollouts(q_net, env, args)

    # Stage 0.2 Sample without replacement from those rollouts and label them (synthetically)
    logging.info('Stage 0.2: Sample without replacement from those rollouts to collect {} labels'.format(args.n_labels_pretraining))
    logging.info('Each label is on a pair of clips of length {}'.format(args.clip_length))
    clip_pairs, rews, mus, mu_counts = sample_and_annotate_clip_pairs(agent_experience, reward_model, args.n_labels_pretraining, args, writers, i_train_round=-1)
    # put labelled clip_pairs into prefs_buffer (also true rewards, just to compute mean/var of true reward across prefs_buffer)
    prefs_buffer.push(clip_pairs, rews, mus)    
    
    # Stage 0.3 Intialise and pretrain reward model
    logging.info('Stage 0.3: Intialise and pretrain reward model for {} batches on those preferences'.format(args.n_epochs_pretrain_rm))
    reward_model = train_reward_model(reward_model, prefs_buffer, optimizer_rm, args, writers[0], i_train_round=-1)

    # evaluate reward model correlation after pretraining
    # if not args.RL_baseline:
    #     test_correlation(reward_model, env, q_net, args, writers[0], i_train_round=-1)

    # BEGIN TRAINING
    for i_train_round in range(args.n_rounds):
        logging.info('[Start Round {}]'.format(i_train_round))
        # Compute mean and variance of true and predicted reward (for normalising rewards sent to agent)
        reward_stats = compute_reward_stats(reward_model, prefs_buffer)
        # Stage 1.1a: Reinforcement learning with (normalised) rewards from current reward model
        q_net, q_target, replay_buffer, agent_experience = do_RL(env, q_net, q_target, optimizer_agent, replay_buffer,
                                                                 reward_model, prefs_buffer, reward_stats, args, writers, i_train_round)
        # Stage 1.1b: Evalulate RL agent performance
        test_returns = test_policy(q_net, reward_model, reward_stats, args)
        log_tested_policy(test_returns, writers, returns_summary, args, i_run, i_train_round)

        # Stage 1.2: Sample clip pairs without replacement from recent rollouts and label them (synthetically)
        num_labels_requested = args.n_labels_per_round[i_train_round]
        logging.info('Stage 1.2: Sample without replacement from those rollouts to collect {} labels/preference tuples'.format(num_labels_requested))
        clip_pairs, rews, mus, mu_counts_round = sample_and_annotate_clip_pairs(agent_experience, reward_model, num_labels_requested, args, writers, i_train_round)
        # put labelled clip_pairs into prefs_buffer and accumulate count of each label acquired
        assert len(clip_pairs) == num_labels_requested
        prefs_buffer.push(clip_pairs, rews, mus)
        mu_counts += mu_counts_round
        
        # Stage 1.3: Train reward model for some epochs on preferences collected to date
        logging.info('Stage 1.3: Train reward model for {} batches on those preferences'.format(args.n_epochs_train_rm))
        reward_model = train_reward_model(reward_model, prefs_buffer, optimizer_rm, args, writers[0], i_train_round)

        # Evaluate reward model correlation (currently not interested in this)
        # if not args.RL_baseline:
        #     test_correlation(reward_model, env, q_net, args, writers[0], i_train_round)

    # log mu_counts for this run
    log_total_mu_counts(mu_counts, writers, args)


def do_RL(env, q_net, q_target, optimizer_agent, replay_buffer, reward_model, prefs_buffer, reward_stats, args, writers, i_train_round):
    writer1, writer2 = writers
    rt_mean, rt_var, rp_mean, rp_var = reward_stats
    # log some info
    if args.RL_baseline:
        logging.info('Stage 1.1: RL using *true reward*')
    else:
        logging.info('Stage 1.1: RL using reward model')
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
        next_state, r_true, _, _ = env.step(action) # one continuous episode
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
                r_pred = reward_model(sa_pair).item() # TODO ensure that this does not effect gradient computation for reward_model in stage 1.3
                dummy_returns['ep']['pred'] += r_pred
                dummy_returns['ep']['pred_norm'] += (r_pred - rp_mean) / np.sqrt(rp_var + 1e-8)
        # prepare for next step
        state = next_state

        # log performance after a "dummy" episode has elapsed
        if (step % args.dummy_ep_length == 0 or step == args.n_agent_train_steps - 1) and step < args.n_agent_train_steps:
            # interpreting writers: 1 == blue == true!
            writer1.add_scalar('3a.dummy_ep_return_against_step/round_{}'.format(i_train_round), dummy_returns['ep']['true'], step)
            writer1.add_scalar('3b.dummy_ep_return_against_step_normalised/round_{}'.format(i_train_round), dummy_returns['ep']['true_norm'], step)
            if not args.RL_baseline:
                writer2.add_scalar('3a.dummy_ep_return_against_step/round_{}'.format(i_train_round), dummy_returns['ep']['pred'], step)
                writer2.add_scalar('3b.dummy_ep_return_against_step_normalised/round_{}'.format(i_train_round), dummy_returns['ep']['pred_norm'], step)
            for key, value in dummy_returns['ep'].items():
                dummy_returns['all'][key].append(value)
                dummy_returns['ep'][key] = 0

        # q_net gradient step
        if step % args.agent_gdt_step_period == 0 and len(replay_buffer) >= 3*q_net.batch_size and step < args.n_agent_train_steps:
            if args.RL_baseline:
                loss_agent = q_learning_loss(q_net, q_target, replay_buffer, args, rt_mean, rt_var)
            else:
                loss_agent = q_learning_loss(q_net, q_target, replay_buffer, args, rp_mean, rp_var, reward_model)
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
       Number of steps taken determined by `args.n_labels_pretraining`. 
       Currently used only in pretraining, but I might refactor s.t. there's
       a single function that I can use for agent-environment
       interaction (with or without training).
    """
    n_initial_steps = args.selection_factor * args.n_labels_pretraining * 2 * args.clip_length
    num_clips       = args.selection_factor * args.n_labels_pretraining * 2
    logging.info('Stage 0.1: Collecting rollouts from untrained policy, {} agent steps'.format(n_initial_steps))
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


def sample_and_annotate_clip_pairs(agent_experience, reward_model, num_labels_requested, args, writers, i_train_round):
    writer1, _ = writers
    writer1.add_scalar('6.labels_requested_per_round', num_labels_requested, i_train_round)
    if args.active_method:
        logging.info('Acquiring clips using {} acquisition function and uncertainty estimates from {}'.format(args.active_method, args.uncert_method))
        if args.acq_search_strategy == 'v0': # TODO refactor acquire_clip_pairs_v0/1
            clip_pairs, rews, mus, label_counts = acquire_clip_pairs_v0(agent_experience, reward_model, num_labels_requested, args, writers, i_train_round)
        elif args.acq_search_strategy == 'v1':
            clip_pairs, rews, mus, label_counts = acquire_clip_pairs_v1(agent_experience, reward_model, num_labels_requested, args, writers, i_train_round)
    else:
        logging.info('Acquiring clips by random acquisition')
        clip_pairs, rews, mus = agent_experience.sample_pairs(num_labels_requested)
        label_counts = log_random_acquisitions(mus, rews, writers, args, i_train_round)
    return clip_pairs, rews, mus, label_counts


def train_reward_model(reward_model, prefs_buffer, optimizer_rm, args, writer1, i_train_round):
    epochs = args.n_epochs_pretrain_rm if i_train_round == -1 else args.n_epochs_train_rm
    reward_model.train() # dropout on
    for epoch in range(epochs):
        with torch.autograd.detect_anomaly():
            clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
            r_hats_batch = reward_model(clip_pair_batch).squeeze(-1) # squeeze the oa_pair dimension that was passed through reward_model
            assert clip_pair_batch.shape == (args.batch_size_rm, 2, args.clip_length, args.obs_act_shape)
            # loss_rm = compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, args.obs_shape, args.act_shape)
            loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
            optimizer_rm.zero_grad()
            loss_rm.backward()
            optimizer_rm.step()
            writer1.add_scalar('7.reward_model_loss/round_{}'.format(i_train_round), loss_rm, epoch)
    return reward_model
            

def test_correlation(reward_model, env, q_net, args, writer1, i_train_round):
    """TODO Work out what dataset we should eval correlation on... currently
       I use the current q_net to generate rollouts and eval on those.
       But this seems bad b/c the dataset changes every round. And indeed,
       correlation seems to go down as training continues, which seems wrong.
    """
    logging.info('Reward model training complete... Evaluating reward model correlation on {} state-action pairs, accumulated on {} rollouts of length {}'.format(
        args.corr_rollout_steps * args.corr_num_rollouts, args.corr_num_rollouts, args.corr_rollout_steps))
    r_xy, plots = eval_rm_correlation(reward_model, env, q_net, args, args.obs_shape, args.act_shape, rollout_steps=args.corr_rollout_steps, num_rollouts=args.corr_num_rollouts)
    log_correlation(r_xy, plots, writer1, round_num=i_train_round)


def compute_reward_stats(reward_model, prefs_buffer):
    """Returns mean and variance of true and predicted reward
       (for normalising rewards sent to agent)
    """
    rt_mean, rt_var = prefs_buffer.compute_mean_var_GT()
    rp_mean, rp_var = compute_mean_var(reward_model, prefs_buffer)
    return rt_mean, rt_var, rp_mean, rp_var