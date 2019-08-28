import torch, gym, gym_barm, pickle, math, argparse
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from q_learning import DQN
from reward_learning import RewardModel, PrefsBuffer, compute_loss_rm
from annotation import AgentExperience, generate_rand_clip_pairing
from tqdm import trange

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--info', type=str, default='', help='Tensorboard log is saved in ./logs/i_run/random_seed/[true|pred]/')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs to repeat the experiment')
    parser.add_argument('--n_rounds', type=int, default=5, help='number of rounds to repeat main training loop')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--env_ID', type=str, default='Gridworld-v0')

    # reward model hyperparamas
    parser.add_argument('--rm_archi', type=str, default='mlp', help='Is reward model an mlp, cnn or cnn_mod?')
    parser.add_argument('--h1_rm', type=int, default=64)
    parser.add_argument('--h2_rm', type=int, default=64)
    parser.add_argument('--h3_rm', type=int, default=None)
    parser.add_argument('--batch_size_rm', type=int, default=16) # same as Ibarz
    parser.add_argument('--lr_rm', type=float, default=1e-4)
    parser.add_argument('--p_dropout_rm', type=float, default=0.5)
    parser.add_argument('--lambda_rm', type=float, default=1e-4, help='coefficient for L2 regularization for reward_model optimization')
    parser.add_argument('--n_epochs_train_rm', type=int, default=5000, help='No. epochs to train reward model per round in main training loop') # Ibarz: 6250
    parser.add_argument('--clip_length', type=int, default=10) #Ibarz/Christiano: 25
    parser.add_argument('--force_label_choice', action='store_true', help='Does synthetic annotator label clips about which it is indifferent as 0.5? If `True`, label equally good clips randomly')
    parser.add_argument('--n_sample_reps', type=int, default=1, help='For debugging: if >1, this will cause n_sample_reps exact copies of the first clip sampled from AgentExperience to be given to acquisition function')

    parser.add_argument('--acq_search_strategy', type=str, default='christiano', help='Whether to use christiano or all_pairs strategy to search for clip pairs. `angelos` is deprecated')
    parser.add_argument('--size_rm_ensemble', type=int, default=1, help='If active_method == ensemble then this must be >= 2')
    parser.add_argument('--selection_factor', type=int, default=1, help='when doing active learning, 1/selection_factor of the randomly sampled clip pairs are sent to human for evaluation')

    # settings that apply only to gridworld
    parser.add_argument('--grid_size', type=int, default=5, help='Length and width of grid')
    parser.add_argument('--n_goals', type=int, default=1)
    parser.add_argument('--n_lavas', type=int, default=1)
    parser.add_argument('--grid_deterministic_reset', action='store_true', help='Do objects in grid reset to same positions once episode terminates?')
    args = parser.parse_args()
    # make cheeky mofifications
    args.env_kwargs = {}
    args.env_kwargs['size']          = args.grid_size
    args.env_kwargs['random_resets'] = not args.grid_deterministic_reset
    args.env_kwargs['n_goals']       = args.n_goals
    args.env_kwargs['n_lavas']       = args.n_lavas
    args.obs_shape = 3*5*5
    args.act_shape = 1
    args.obs_act_shape = 3*5*5 + 1
    args.n_actions = 4
    args.acquistion_func = lambda x : x
    return args


def get_test_data(env, n_clips_total, n_labels_total, args):
    agent_experience_test = collect_random_experience(env, n_clips_total, args)
    # carve up test experience into clip pairs (with corresponding true rews and mus)
    clip_pairs_test_, _, mus_test_ = generate_rand_clip_pairing(agent_experience_test, n_labels_total, args)
    # make them into tensors
    return torch.from_numpy(clip_pairs_test_).float(), torch.from_numpy(mus_test_).float()


def test_rm():
    # setup
    args = parse_arguments()
    args.logdir = './logs/{}/{}'.format(args.info, args.random_seed)
    writer1 = SummaryWriter(log_dir=args.logdir+'/train')
    writer2 = SummaryWriter(log_dir=args.logdir+'/train_lower')
    writer3 = SummaryWriter(log_dir=args.logdir+'/test')
    writer4 = SummaryWriter(log_dir=args.logdir+'/test_lower')
    env = gym.make(args.env_ID, **args.env_kwargs)
    env.seed(args.random_seed)
    obs = env.reset()
    args.oa_dtype = obs.dtype
    # create train and test data from random policy
    n_labels_total = 500
    n_clips_total = 2 * n_labels_total
    agent_experience_train = collect_random_experience(env, n_clips_total, args)
    # carve up train experience into clip pairs and push onto prefs buffer for sampling from
    clip_pairs_train_all, _, mus_train_all = generate_rand_clip_pairing(agent_experience_train, n_labels_total, args)
    # get TEST data
    clip_pairs_TEST, mus_TEST = get_test_data(env, n_clips_total, n_labels_total, args)
    assert clip_pairs_TEST.shape == (n_labels_total, 2, args.clip_length, args.obs_act_shape)
    assert mus_TEST.shape == (n_labels_total,)
    # compute lower bound for test loss (relative to batch size of args.batch_size_rm, since we use reduction='sum')
    n_indifferent_labels_TEST = Counter(mus_TEST.numpy()).get(0.5, 0)
    loss_lower_bound_TEST = n_indifferent_labels_TEST * math.log(2) / n_labels_total * args.batch_size_rm

    for n_labels in range(50, 501, 50):
        n_clips = 2 * n_labels
        # initialise reward model and optimizer
        reward_model = RewardModel(args.obs_shape, args.act_shape, args)
        optimizer_rm = optim.Adam(reward_model.parameters(), lr=args.lr_rm, weight_decay=args.lambda_rm)
        # get n_labels to train on
        assert n_labels_total == len(clip_pairs_train_all)
        idx = np.random.choice(n_labels_total, size=n_labels, replace=False)
        clip_pairs_train, mus_train = clip_pairs_train_all[idx], mus_train_all[idx]
        assert clip_pairs_train.shape == (n_labels, 2, args.clip_length, args.obs_act_shape)
        assert mus_train.shape == (n_labels, )
        # train reward model!
        reward_model.train()
        for epoch in trange(args.n_epochs_train_rm, desc='Training reward model for {} epochs, with {} labels'.format(args.n_epochs_train_rm, n_labels)):
            # draw a minibatch
            idx = np.random.choice(n_labels, size=args.batch_size_rm, replace=False)
            clip_pairs_batch_, mus_batch_ = clip_pairs_train[idx], mus_train[idx]
            clip_pairs_batch  = torch.from_numpy(clip_pairs_batch_).float()
            mus_batch         = torch.from_numpy(mus_batch_).float()
            # pass clip pairs thru reward model to get predicted rewards
            r_hats_batch = reward_model(clip_pairs_batch, mode='clip_pair_batch', normalise=False).squeeze(-1)
            # compute loss
            loss_rm = compute_loss_rm(r_hats_batch, mus_batch)
            # backprop
            optimizer_rm.zero_grad()
            loss_rm.backward()
            optimizer_rm.step()
            writer1.add_scalar('reward_model_loss/label_{}'.format(n_labels), loss_rm, epoch)
            # compute lower bound for loss_rm and plot this too
            n_indifferent_labels = Counter(mus_batch.numpy()).get(0.5, 0)
            loss_lower_bound = n_indifferent_labels * math.log(2)
            writer2.add_scalar('reward_model_loss/label_{}'.format(n_labels), loss_lower_bound, epoch)
            # compute test loss every so often (6 times per training)
            # we compute over entire test set of 500 labels
            if epoch % 1000 == 0 or epoch == args.n_epochs_train_rm - 1:
                reward_model.eval() # turn dropout off for computing test loss
                # pass clip pairs thru reward model to get predicted rewards
                r_hats_TEST = reward_model(clip_pairs_TEST, mode='clip_pair_batch', normalise=False).squeeze(-1)
                # compute loss
                loss_rm_TEST = compute_loss_rm(r_hats_TEST, mus_TEST).detach()  / n_labels_total * args.batch_size_rm
                # log them
                writer3.add_scalar('reward_model_loss/label_{}'.format(n_labels), loss_rm, epoch)
                # log lower bound too
                writer4.add_scalar('reward_model_loss/label_{}'.format(n_labels), loss_lower_bound, epoch)
                reward_model.train() # turn dropout back on


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


if __name__ == '__main__':
    test_rm()