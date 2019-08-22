import torch, gym, gym_barm
from q_learning import CnnDQN

class Args:
    """
    Pseudo arguments for reconstructing CnnDQN
    """
    def __init__(self):
        self.obs_shape = 3*84*84
        self.n_actions = 4
        self.batch_size_agent = 32
        self.gamma = 0.99
        self.target_update_tau = 0.001
        self.env_kwargs = {'partial': False, 'size': 5, 'random_resets': True, 'terminate_ep_if_done': True}
        self.random_seed = 0

def show_video(n_episodes=5, n_render_steps_per_ep=10):
    args = Args()
    q_net = CnnDQN(args.obs_shape, args.n_actions, args)
    checkpoint = torch.load('./logs_old/in_progress/more_failure/sam2/RandAcq_dones-1e-4-mod/0/checkpts/agent/0-14.pt')
    q_net.load_state_dict(checkpoint['policy_state_dict'])
    # i_round = checkpoint['round']
    # test_no = checkpoint['i_sub_round']
    q_net.eval()

    env = gym.make('Gridworld-v0', **args.env_kwargs)
    env.seed(args.random_seed)
    state = env.reset()
    env.render() # this will hang until window is closed
    
    n = 0
    step = 0
    returns = {'ep': 0, 'all': []}
    while n < n_episodes:
        # agent interact with env
        action = q_net.act(state)
        next_state, r_true, done, _ = env.step(action)
        if step <= n_render_steps_per_ep: env.render(mode='nonblock') # render the first 5 steps
        returns['ep'] += r_true # record step info
        # prepare for next step
        if done:
            returns['all'].append(returns['ep'])
            returns['ep'] = 0
            next_state = env.reset()
            n += 1
            step = 0
        state = next_state
        step += 1
    print('Returns: {}'.format(returns['all']))

show_video()