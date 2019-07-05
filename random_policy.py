import numpy as np
from tqdm import trange

def do_random_experiment(env, args, writer1, writer2):
    for i_train_round in args.n_rounds:
        dummy_returns = {'ep': 0, 'all': []}
        env.reset()
        for step in trange(args.n_agent_steps, desc='Taking random actions for {} steps'.format(args.n_agent_steps), dynamic_ncols=True):
            # agent interact with env
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            _, r_true, _, _ = env.step(action) # one continuous episode
            dummy_return += r_true # record step info

            # log performance after a "dummy" episode has elapsed
            if (step % args.dummy_ep_length == 0 or step == args.n_agent_steps - 1):
                writer2.add_scalar('dummy ep return against step/round {}'.format(i_train_round), dummy_returns['ep'], step)
                dummy_returns['all'].append(dummy_returns['ep'])
                dummy_returns['ep'] = 0

        # log mean recent return this training round
        mean_dummy_true_returns = np.sum(np.array(dummy_returns['all'][-3:])) / 3. # 3 dummy eps is the final 3*200/2000 == 3/10 eps in the round
        writer2.add_scalar('mean dummy ep returns per training round', mean_dummy_true_returns, i_train_round)