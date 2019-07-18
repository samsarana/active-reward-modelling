"""Components of Deep RL from Human Preferences training protocol"""

import logging
from q_learning import *
from reward_learning import *
from active_learning import *
from test_policy import *

def do_random_experiment(env, args, returns_summary, writers, i_run):
    writer1, _, _ = writers
    for i_train_round in range(args.n_rounds):
        logging.info('[Start Round {}]'.format(i_train_round))
        dummy_returns = {'ep': 0, 'all': []}
        env.reset()
        logging.info('Taking random actions for {} steps'.format(args.n_agent_steps))
        for step in range(args.n_agent_steps):
            # agent interact with env
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            _, r_true, _, _ = env.step(action) # one continuous episode
            dummy_returns['ep'] += r_true # record step info

            # log performance after a "dummy" episode has elapsed
            if (step % args.dummy_ep_length == 0 or step == args.n_agent_steps - 1):
                writer1.add_scalar('3a.dummy_ep_return_against_step/round_{}'.format(i_train_round), dummy_returns['ep'], step)
                dummy_returns['all'].append(dummy_returns['ep'])
                dummy_returns['ep'] = 0

        # log mean recent return this training round
        # mean_dummy_true_returns = np.sum(np.array(dummy_returns['all'][-3:])) / 3. # 3 dummy eps is the final 3*200/2000 == 3/10 eps in the round
        mean_dummy_true_returns = np.sum(np.array(dummy_returns['all'])) / len(dummy_returns['all'])
        writer1.add_scalar('2.dummy_mean_ep_returns_per_training_round', mean_dummy_true_returns, i_train_round)
        test_and_log_random_policy(writers, returns_summary, args, i_run, i_train_round)


def do_RL(env, q_net, q_target, optimizer_agent, replay_buffer, reward_model, prefs_buffer, reward_stats, args, writers, i_train_round):
    writer1, writer2, _ = writers
    rt_mean, rt_var, rp_mean, rp_var = reward_stats
    # Prepare to train agent for args.n_agent_steps
    # (or if active_method, collect more experience but train same amount and on same experience)
    if args.active_method:
        n_agent_steps = args.selection_factor * args.n_agent_steps
        n_train_steps = args.n_agent_steps
        logging.info('Active Learning so will take {} further steps *without training*; this goes into agent_experience, so algo can sample extra *possible* clip pairs, and keep the best 1/{}'.format(n_agent_steps - n_train_steps, args.selection_factor))
        logging.info('Stage 1.1: RL using reward model for {} agent steps, of which the first {} include training'.format(n_agent_steps, n_train_steps))
    else:
        n_agent_steps = n_train_steps = args.n_agent_steps
        if args.RL_baseline:
            logging.info('Stage 1.1: RL using *true reward* for {} agent steps'.format(n_agent_steps))
        else:
            logging.info('Stage 1.1: RL using reward model for {} agent steps'.format(n_agent_steps))
    num_clips = int(n_agent_steps//args.clip_length)
    assert n_agent_steps % args.clip_length == 0
    agent_experience = AgentExperience((num_clips, args.clip_length, args.obs_act_shape), args.force_label_choice) # since episodes do not end we will collect one long trajectory then sample clips from it     
    dummy_returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
                    'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}
    # train!
    state = env.reset()
    for step in range(n_agent_steps):
        # agent interact with env
        action = q_net.act(state, q_net.epsilon)
        assert env.action_space.contains(action)
        next_state, r_true, _, _ = env.step(action) # one continuous episode
        # record step info
        sa_pair = torch.tensor(np.append(state, action)).float()
        agent_experience.add(sa_pair, r_true) # include reward in order to later produce synthetic prefs
        if step < n_train_steps:
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
        if (step % args.dummy_ep_length == 0 or step == args.n_agent_steps - 1) and step < n_train_steps:
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
        if step % args.agent_gdt_step_period == 0 and len(replay_buffer) >= 3*q_net.batch_size and step < n_train_steps:
            if args.RL_baseline:
                loss_agent = q_learning_loss(q_net, q_target, replay_buffer, rt_mean, rt_var)
            else:
                loss_agent = q_learning_loss(q_net, q_target, replay_buffer, rp_mean, rp_var, reward_model)
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
            if step % args.target_update_period == 0: # soft update target parameters
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


def do_pretraining(env, q_net, reward_model, prefs_buffer, args, writers):
    writer1, _, _ = writers
    # Stage 0.1 Initialise policy and do some rollouts
    epsilon_pretrain = 0.1 # for now I'll use a constant epilson during pretraining
    # n_initial_steps = args.n_initial_agent_steps
    n_initial_steps = args.n_labels_pretraining * 2 * args.clip_length
    if args.active_method:
        n_initial_steps *= args.selection_factor
        logging.info('Doing Active Learning ({} method), so collect {}x more rollouts than usual'.format(
                args.active_method, args.selection_factor))
    num_clips = int(n_initial_steps//args.clip_length)
    assert n_initial_steps % args.clip_length == 0, "Agent should take a number of steps that's divisible by the desired clip_length"
    agent_experience = AgentExperience((num_clips, args.clip_length, args.obs_act_shape), args.force_label_choice)
    state = env.reset()
    logging.info('Stage 0.1: Collecting rollouts from untrained policy, {} agent steps'.format(n_initial_steps))
    for _ in range(n_initial_steps):
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
    logging.info('Stage 0.2: Sample without replacement from those rollouts to collect {} labels. Each label is on a pair of clips of length {}'.format(args.n_labels_pretraining, args.clip_length))
    writer1.add_scalar('6.labels_requested_per_round', args.n_labels_pretraining, -1)
    if args.active_method:
        if args.acq_search_strategy == 'v0':
            clip_pairs, rews, mus, mu_counts, rand_mu_counts = acquire_clip_pairs_v0(agent_experience, reward_model, args.n_labels_pretraining, args, writers, i_train_round=-1)
        elif args.acq_search_strategy == 'v1':
            clip_pairs, rews, mus, mu_counts, rand_mu_counts = acquire_clip_pairs_v1(agent_experience, reward_model, args.n_labels_pretraining, args, writers, i_train_round=-1)
    else:
        clip_pairs, rews, mus = agent_experience.sample_pairs(args.n_labels_pretraining)
        mu_counts, rand_mu_counts = log_random_acquisitions(mus, rews, writers, args, round_num=-1), None
    # put chosen clip_pairs, true rewards (just to compute mean/var of true reward across prefs_buffer)
    # and synthetic preferences into prefs_buffer
    prefs_buffer.push(clip_pairs, rews, mus)
    
    # Stage 0.3 Intialise and pretrain reward model
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm)
    reward_model.train() # dropout on
    logging.info('Stage 0.3: Intialise and pretrain reward model for {} batches on those preferences'.format(args.n_epochs_pretrain_rm))
    for epoch in range(args.n_epochs_pretrain_rm):
        with torch.autograd.detect_anomaly(): # detects NaNs; useful for debugging
            clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
            r_hats_batch = reward_model(clip_pair_batch).squeeze(-1)
            loss_rm = compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, args.obs_shape, args.act_shape)
            # TODO call clean version instead i.e. loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
            reward_model.train() # dropout
            optimizer_rm.zero_grad()
            loss_rm.backward()
            optimizer_rm.step()
            writer1.add_scalar('7.reward_model_loss/pretraining', loss_rm, epoch)

    # evaluate reward model correlation after pretraining
    # if not args.RL_baseline:
    #     logging.info('Reward model training complete... Evaluating reward model correlation on {} state-action pairs, accumulated on {} rollouts of length {}'.format(
    #             args.corr_rollout_steps * args.corr_num_rollouts, args.corr_num_rollouts, args.corr_rollout_steps))
    #     r_xy, plots = eval_rm_correlation(reward_model, env, q_net, args, args.obs_shape, args.act_shape, rollout_steps=args.corr_rollout_steps, num_rollouts=args.corr_num_rollouts)
    #     log_correlation(r_xy, plots, writer1, round_num=-1)

    return reward_model, prefs_buffer, mu_counts, rand_mu_counts


def do_training(env, q_net, q_target, reward_model, prefs_buffer, args, writers, returns_summary, i_run, total_mu_counts, total_rand_mu_counts):
    writer1, _ = writers
    # Stage 1.0: Setup
    optimizer_agent = optim.Adam(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent) # q_net initialised above
    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm) # reinitialise optimizer so we don't need to pass it between funcs

    for i_train_round in range(args.n_rounds):
        logging.info('[Start Round {}]'.format(i_train_round))
        # Stage 1.1a: Reinforcement learning with (normalised) rewards from current reward model
        # compute mean and variance of true and predicted reward (for normalising rewards sent to agent)
        rt_mean, rt_var = prefs_buffer.compute_mean_var_GT()
        rp_mean, rp_var = compute_mean_var(reward_model, prefs_buffer)
        reward_stats = (rt_mean, rt_var, rp_mean, rp_var)
        q_net, q_target, replay_buffer, agent_experience = do_RL(env, q_net, q_target, optimizer_agent, replay_buffer,
                                                                 reward_model, prefs_buffer, reward_stats, args, writers, i_train_round)
        # Stage 1.1b: Evalulate RL agent performance
        test_returns = test_policy(q_net, reward_model, reward_stats, args, random_seed=i_run)
        log_tested_policy(test_returns, writers, returns_summary, args, i_run, i_train_round)

        # Stage 1.2: Sample clip pairs without replacement from recent rollouts and label them (synthetically)
        # num_labels_requested = int(50*5 / (i_train_round + 5)) #int(58.56 * (5e6 / (i_train_round * args.n_agent_steps + 5e6) )) # compute_label_annealing_const.py
        num_labels_requested = args.n_labels_per_round[i_train_round]
        logging.info('Stage 1.2: Sample without replacement from those rollouts to collect {} labels/preference tuples'.format(num_labels_requested))
        writer1.add_scalar('6.labels_requested_per_round', num_labels_requested, i_train_round)
        if args.active_method:
            if args.acq_search_strategy == 'v0': # TODO refactor acquire_clip_pairs_v0/1
                clip_pairs, rews, mus, mu_counts, rand_mu_counts = acquire_clip_pairs_v0(agent_experience, reward_model, num_labels_requested, args, writers, i_train_round)
            elif args.acq_search_strategy == 'v1':
                clip_pairs, rews, mus, mu_counts, rand_mu_counts = acquire_clip_pairs_v1(agent_experience, reward_model, num_labels_requested, args, writers, i_train_round)
            total_rand_mu_counts += rand_mu_counts
        else:
            clip_pairs, rews, mus = agent_experience.sample_pairs(num_labels_requested)
            mu_counts = log_random_acquisitions(mus, rews, writers, args, i_train_round)
        total_mu_counts += mu_counts
        # put labelled clip_pairs into prefs_buffer
        assert len(clip_pairs) == num_labels_requested
        prefs_buffer.push(clip_pairs, rews, mus)
        
        # Stage 1.3: Train reward model
        reward_model.train() # dropout on
        logging.info('Stage 1.3: Train reward model for {} batches on those preferences'.format(args.n_epochs_train_rm))
        for epoch in range(args.n_epochs_train_rm):
            with torch.autograd.detect_anomaly():
                clip_pair_batch, mu_batch = prefs_buffer.sample(args.batch_size_rm)
                r_hats_batch = reward_model(clip_pair_batch).squeeze(-1) # squeeze the oa_pair dimension that was passed through reward_model
                assert clip_pair_batch.shape == (args.batch_size_rm, 2, args.clip_length, args.obs_act_shape)
                loss_rm = compute_loss_rm_wchecks(r_hats_batch, mu_batch, args, args.obs_shape, args.act_shape)
                # TODO call clean version instead i.e. loss_rm = compute_loss_rm(r_hats_batch, mu_batch)
                optimizer_rm.zero_grad()
                loss_rm.backward()
                optimizer_rm.step()
                writer1.add_scalar('7.reward_model_loss/round_{}'.format(i_train_round), loss_rm, epoch)

        # evaluate reward model correlation
        # if not args.RL_baseline:
        #     logging.info('Reward model training complete... Evaluating reward model correlation on {} state-action pairs, accumulated on {} rollouts of length {}'.format(
        #         args.corr_rollout_steps * args.corr_num_rollouts, args.corr_num_rollouts, args.corr_rollout_steps))
        #     r_xy, plots = eval_rm_correlation(reward_model, env, q_net, args, args.obs_shape, args.act_shape, rollout_steps=args.corr_rollout_steps, num_rollouts=args.corr_num_rollouts)
        #     log_correlation(r_xy, plots, writer1, round_num=i_train_round)