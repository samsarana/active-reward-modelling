import logging
import numpy as np
from time import time
from q_learning import *
from test_policy import *
from rl_logging import *
import gym
from utils import one_hot_action

def do_RL(env, q_net, q_target, optimizer_agent, replay_buffer, epsilon_schedule,
          agent_experience, reward_model, returns_summary,
          true_reward_stats, args, writers, i_run, i_train_round):
    writer1, writer2 = writers
    train_returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
                    'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}
    is_saving_video, done_saving_video = False, False
    start_step = 0 if args.reinit_agent else args.n_agent_steps * (i_train_round - 1)
    n_episodes = 0
    i_test = 0
    state = env.reset()
    for step in range(start_step, start_step + args.n_agent_steps):
        # agent interact with env
        epsilon = epsilon_schedule.value(step)
        action = q_net.act(state, epsilon)
        assert env.action_space.contains(action)
        next_state, r_true, done, _ = env.step(action) # one continuing episode
        # record step info
        # sa_pair = torch.tensor(np.append(state, action)).float()
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_one_hot = one_hot_action(action, env)
        sa_pair = np.append(state, action_one_hot).astype(args.oa_dtype, casting='unsafe') # in case len(state.shape) > 1 (gridworld, atari), np.append will flatten it
        assert (sa_pair == np.append(state, action_one_hot)).all() # check casting done safely. should be redundant since i set oa_dtype based on env, earlier. but you can never be too careful since this would fail silently!
        if not args.RL_baseline or args.normalise_rewards:
            agent_experience.add(sa_pair, r_true) # include reward in order to later produce synthetic prefs
        if args.agent_gets_dones:
            replay_buffer.push(state, action, r_true, next_state, done)
        else:
            replay_buffer.push(state, action, r_true, next_state, False) # done=False since agent thinks the task is continual; r_true used only when args.RL baseline
        train_returns = log_agent_step(sa_pair, r_true, train_returns, true_reward_stats, reward_model, args)
        # prepare for next step
        state = next_state
        if done:
            if args.save_video and not (is_saving_video or done_saving_video) and (args.n_agent_steps_before_test - step < 4*env.spec.max_episode_steps):
                # save the final 4ish train episodes (see https://github.com/openai/gym/wiki/FAQ#how-do-i-export-the-run-to-a-video-file)
                env = wrappers.Monitor(env, args.logdir + '/videos/train/' + str(time()) + '/')
                is_saving_video = True # don't call the env wrapper again
            state = env.reset()
            train_returns = log_agent_episode(train_returns, writers, step, i_train_round, args)
            n_episodes += 1

            # q_net gradient step
        # if step >= args.agent_learning_starts and step % args.agent_gdt_step_period == 0 and \
        #         len(replay_buffer) >= 3*args.batch_size_agent:
            if step >= args.agent_learning_starts and len(replay_buffer) >= 3*args.batch_size_agent: # we now make learning updates at the end of every episode
                if args.RL_baseline:
                    loss_agent = q_learning_loss(q_net, q_target, replay_buffer, args, normalise_rewards=args.normalise_rewards, true_reward_stats=true_reward_stats)
                else:
                    loss_agent = q_learning_loss(q_net, q_target, replay_buffer, args, reward_model=reward_model, normalise_rewards=args.normalise_rewards)
                optimizer_agent.zero_grad()
                loss_agent.backward()
                optimizer_agent.step()
                writer1.add_scalar('7.agent_loss/round_{}'.format(i_train_round), loss_agent, step)
                # scheduler.step() # Ibarz doesn't mention lr annealing...
                writer1.add_scalar('8.agent_epsilon/round_{}'.format(i_train_round), epsilon, step)
            if args.epsilon_annealing_scheme == 'exp':
                epsilon_schedule.step()

        # update q_target
        if step % args.target_update_period == 0: # update target parameters
            for target_param, local_param in zip(q_target.parameters(), q_net.parameters()):
                target_param.data.copy_(q_net.tau*local_param.data + (1.0-q_net.tau)*target_param.data)

        if step > 0 and step % args.agent_test_period == 0 or step == start_step + args.n_agent_steps - 1:
            # Evalulate RL agent performance
            logging.info("Agent has taken {} steps. Testing performance for 100 episodes".format(step))
            test_returns = test_policy(q_net, reward_model, true_reward_stats, args, writers, i_train_round, i_test)
            mean_test_ret_true = log_agent_test(train_returns, test_returns, returns_summary, step, i_test, i_train_round, i_run, writers, args)
            # save current policy
            save_policy(q_net, optimizer_agent, i_train_round, i_test, args)
            # Possibly end training if mean_test_ret_true is above the threshold
            if not args.continue_once_solved and env.spec.reward_threshold != None and mean_test_ret_true >= env.spec.reward_threshold:
                writer1.add_scalar('10.n_episodes', n_episodes, i_train_round)
                raise SystemExit("Environment solved, moving onto next run.")
            # reset dict logging returns since last test
            train_returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
                    'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}
            i_test += 1

    writer1.add_scalar('10.n_episodes', n_episodes, i_train_round)
    return q_net, q_target, replay_buffer, agent_experience