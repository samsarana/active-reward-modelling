import numpy as np
import torch, gym, gym_barm, pickle, math
from collections import Counter
from q_learning import DQN
from reward_learning import RewardModel, PrefsBuffer, compute_loss_rm
from annotation import AgentExperience, generate_rand_clip_pairing
from utils import one_hot_action

class Args:
    """
    Pseudo arguments for reconstructing
    DQN and reward model
    """
    def __init__(self):
        # env settings
        self.env_ID = 'Gridworld-v0'
        self.env_kwargs = {'size': 4, 'random_resets': True, 'n_lavas': 1}
        self.random_seed = 0
        self.obs_shape = 3*4*4
        self.act_shape = 4
        self.obs_act_shape = 3*4*4 + 4
        self.n_actions = 4
        self.oa_dtype = None
        # agent settings
        self.h1_agent = 256
        self.h2_agent = 256
        self.h3_agent = 512
        self.batch_size_agent = 32
        self.gamma = 0.99
        self.target_update_tau = 0.001
        # reward model settings
        self.h1_rm = 64
        self.h2_rm = 64
        self.h3_rm = None
        self.p_dropout_rm = 0.5
        self.clip_length = 10
        self.prefs_buffer_size = None
        self.force_label_choice = False
        self.n_sample_reps = 1
        self.acq_search_strategy = 'christiano'
        self.selection_factor = 1
        self.acquistion_func = lambda x : x
    

def load_objects(args):
    # checkpoint_dir = './logs_old/in_progress/sam2/RandAcq/0/checkpts/'
    checkpoint_dir = './logs_old/in_progress/sam2/RandAcq/0/checkpts/'
    checkpoint_agent    = torch.load(checkpoint_dir + 'agent/10-4.pt')
    checkpoint_rm       = torch.load(checkpoint_dir + 'rm/rm0-round9.pt')
    # checkpoint_prefs    = torch.load(checkpoint_dir + 'prefs/buff-9.pkl')
    # checkpoint_stats    = torch.load(checkpoint_dir + 'prefs/stats-9.pkl')
    # load DQN
    q_net = DQN(args.obs_shape, args.n_actions, args)
    q_net.load_state_dict(checkpoint_agent['policy_state_dict'])
    q_net.eval()
    # load reward model
    reward_model = RewardModel(args.obs_shape, args.act_shape, args)
    reward_model.load_state_dict(checkpoint_rm['rm_state_dict'])
    reward_model.eval()
    # load prefs_buffer and true_reward_stats
    # with open(checkpoint_prefs, 'rb') as in_path:
    #     prefs_buffer = pickle.load(in_path)
    # with open(checkpoint_stats, 'rb') as in_path:
    #     true_reward_stats = pickle.load(in_path)
    return q_net, reward_model#, prefs_buffer, true_reward_stats


def collect_random_experience(env, n_clips, args):
    agent_experience = AgentExperience(n_clips, args)
    n_steps = n_clips * args.clip_length
    state, n_episodes = env.reset(), 0
    for step in range(n_steps):
        action = env.action_space.sample()
        next_state, r_true, done, _ = env.step(action) # one continuing episode
        # record step info
        sa_pair = np.append(state, action).astype(args.oa_dtype, casting='unsafe') # in case len(state.shape) > 1 (gridworld, atari), np.append will flatten it
        assert (sa_pair == np.append(state, action)).all() # check casting done safely. should be redundant since i set oa_dtype based on env, earlier. but you can never be too careful since this would fail silently!
        agent_experience.add(sa_pair, r_true) # include reward in order to later produce synthetic prefs
        # prepare for next step
        state = next_state
        if done:
            n_episodes += 1
            state = env.reset()
    print('Finished {} random steps, {} episodes'.format(n_steps, n_episodes))
    return agent_experience


def compute_rm_validation_loss():
    # setup
    args = Args()
    env = gym.make(args.env_ID, **args.env_kwargs)
    env.seed(args.random_seed)
    obs = env.reset()
    args.oa_dtype = obs.dtype
    # specifify over how many labelled clip pairs we want to compute validation loss
    n_labels = 1000
    n_clips = 2 * n_labels
    # load particular reward model specified in load_objects()
    checkpoint_rm  = torch.load('./logs_old/in_progress/sam1/RandAcq/0/checkpts/' + 'rm/2.pt')
    # load reward model
    reward_model = RewardModel(args.obs_shape, args.act_shape, args)
    reward_model.load_state_dict(checkpoint_rm['rm_state_dict'])
    reward_model.eval()
    # collect random experience
    agent_experience = collect_random_experience(env, n_clips, args)
    # carve up all of that experience into clip pairs (with corresponding true rews and mus)
    clip_pairs, rews, mus = generate_rand_clip_pairing(agent_experience, n_labels, args) # I've never used up all of agent experience. will this be ok?
    # push them on prefs buffer
    args.prefs_buffer_size = n_labels
    prefs_buffer = PrefsBuffer(args)
    prefs_buffer.push(clip_pairs, rews, mus)
    # get them all out of the prefs buffer (this might be a strain on memory?)
    clip_pair_batch, mu_batch = prefs_buffer.sample(prefs_buffer.current_length)
    # pass clip pairs thru reward model to get predicted rewards
    r_hats_batch = reward_model(clip_pair_batch, mode='clip_pair_batch', normalise=False).squeeze(-1)
    # compute loss
    valid_loss_rm = compute_loss_rm(r_hats_batch, mu_batch).detach().item()
    # compute lower bound for loss
    n_indifferent_labels = Counter(mu_batch.numpy()).get(0.5, 0)
    loss_lower_bound = n_indifferent_labels * math.log(2)
    print('Printing losses for batch size of 16:')
    print('Validation loss: {:.2f}'.format(valid_loss_rm / n_labels * 16))
    print('Lower bound:     {:.2f}'.format(loss_lower_bound / n_labels * 16))
    import ipdb; ipdb.set_trace()

def show_video(n_episodes=5, n_render_steps_per_ep=10, mode='agent'):
    args = Args()
    letters_to_actions = {
        'w': 0,
        's': 1,
        'a': 2,
        'd': 3
    }
    # q_net, reward_model, prefs_buffer, true_reward_stats = load_objects(args)
    q_net, reward_model = load_objects(args)
    # create environment
    env = gym.make(args.env_ID, **args.env_kwargs)
    env.seed(args.random_seed)
    state = env.reset()
    if mode == 'agent': env.render() # this will hang until window is closed
    # do some rollouts!
    n = 0
    step = 0
    returns = {'ep': {'true': 0, 'pred': 0, 'pred_norm': 0}, 'all': {'true': [], 'pred': [], 'pred_norm': []}}
    ep_lens = []
    while n < n_episodes:
        if mode == 'agent' and step <= n_render_steps_per_ep:
            env.render(mode='nonblock') # render the first steps of the episode
        elif mode == 'human':
            env.render()
        if mode == 'agent':
            action = q_net.act(state)
        else:
            is_valid_key = False
            while not is_valid_key:
                key = input('Input action (w,a,s,d): ')
                if key == 'q': raise SystemExit("User ended game")
                try:
                    action = letters_to_actions[key]
                except KeyError:
                    pass
                else:
                    is_valid_key = True
        next_state, r_true, done, _ = env.step(action)
        # compute predicted reward
        action_one_hot = one_hot_action(action, env)
        sa_pair = torch.from_numpy(np.append(state, action_one_hot)).float()
        assert sa_pair.shape == (args.obs_shape + args.act_shape,)
        r_pred = reward_model(sa_pair).detach().item()
        r_pred_norm = reward_model(sa_pair, normalise=True).detach().item()
        # record step info
        print('True reward: {}'.format(r_true)) # hmm, i wonder what true normalised rewards are. DQN RL baseline performance was improved greatly by normalising in exactly the same was as I do for reward modelling. this seems to warrant more investigation. though DQN doesn't seem like the bottleneck right now
        print('Pred reward: {:.2f}'.format(r_pred))
        print('-normalised: {:.2f}'.format(r_pred_norm))
        returns['ep']['true'] += r_true
        returns['ep']['pred'] += r_pred
        returns['ep']['pred_norm'] += r_pred_norm
        # prepare for next step
        if done:
            if mode == 'human':
                print('Episode done!')
                env.render()
            for reward_type, value in returns['ep'].items():
                returns['all'][reward_type].append(value)
                returns['ep'][reward_type] = 0
            ep_lens.append(step)
            next_state = env.reset()
            n += 1
            step = 0
            print('\nStart episode {}\n'.format(n))
        state = next_state
        step += 1

    # print summary
    print('Showing returns for every episode')
    for return_type, rets in returns['all'].items():
        print('\{}:'.format(return_type))
        for ret in rets:
            print('{:2f}'.format(ret), end='  ')
    print('\nEpisode lengths')
    for ep_len in ep_lens:
        print('{}'.format(ep_len), end='  ')
    # make a test set and compute loss (as well as lower bound. reuse your code i.e. just add to prefs buffer, that'll do all the mu calculation etc; just need to eval batch not minibatch loss!)


if __name__ == '__main__':
    # compute_rm_validation_loss()
    show_video(mode='human')