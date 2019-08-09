def cartpole_defaults(args):
    """NB to truly recover my old defaults,
       should set n_agent_steps := 30000
       and agent_learning_start := 27000
       Even then, there's still a slight difference
       because the extra steps are now
       taken at beginning of training. But I think
       this isn't an important moving part.
    """
    args.env_str = 'cartpole'
    args.lr_agent = 1e-3
    args.replay_buffer_size = 30000
    args.target_update_period = 1
    args.target_update_tau = 8e-2
    args.agent_gdt_step_period = 1
    args.gamma = 0.9
    args.epsilon_stop = 0.01
    args.exploration_fraction = 0.1
    # args.n_labels_per_round = 1
    args.n_agent_steps = 8000 # see docstr
    args.agent_test_frequency = 2
    args.agent_learning_starts = 0 # see docstr
    args.reinit_agent = True
    args.n_epochs_pretrain_rm = 2000
    args.n_epochs_train_rm = 2000
    args.clip_length = 25
    args.reinit_rm = True
    args.n_rounds = 50
    args.n_runs = 20
    return args


def acrobot_sam_defaults(args):
    args.lr_agent = 1e-3
    args.replay_buffer_size = 30000
    args.target_update_period = 1
    args.target_update_tau = 8e-2
    args.agent_gdt_step_period = 1
    args.gamma = 0.99
    args.epsilon_stop = 0.01
    args.exploration_fraction = 0.1
    # args.n_labels_per_round = 1
    args.n_agent_steps = 100000
    args.agent_test_frequency = 20 # 100k / 20 = 5k
    args.agent_learning_starts = 0
    args.reinit_agent = True
    args.n_epochs_pretrain_rm = 2000 # not yet tested
    args.n_epochs_train_rm = 2000 # not yet tested
    args.clip_length = 25
    args.reinit_rm = True
    return args


def openai_defaults(args):
    """NB These settings succeeded for Acrobot
       though not quite as well as acrobot_sam_defaults
    """
    args.lr_agent = 5e-4
    args.replay_buffer_size = 50000
    args.target_update_period = 500
    args.target_update_tau = 1
    args.agent_gdt_step_period = 1
    args.gamma = 1.0
    args.epsilon_stop = 0.02
    args.exploration_fraction = 0.1
    # args.n_labels_per_round = 1
    args.n_agent_steps = 100000
    args.agent_test_frequency = 20 # 100k / 20 = 5k
    args.agent_learning_starts = 1000
    args.reinit_agent = True
    args.n_epochs_pretrain_rm = 2000 # not yet tested
    args.n_epochs_train_rm = 2000 # not yet tested
    args.clip_length = 25
    args.reinit_rm = True
    return args


def openai_atari_defaults(args):
    args.lr_agent = 1e-4
    args.replay_buffer_size = 10000
    args.target_update_period = 1000
    args.target_update_tau = 1
    args.agent_gdt_step_period = 4
    args.gamma = 0.99
    args.epsilon_stop = 0.01
    args.exploration_fraction = 0.1
    # args.n_labels_per_round = 1
    args.n_agent_steps = 100000
    args.agent_test_frequency = 20 # 100k / 20 = 5k
    args.agent_learning_starts = 10000
    args.reinit_agent = True
    args.n_epochs_pretrain_rm = 2000 # not yet tested
    args.n_epochs_train_rm = 2000 # not yet tested
    args.clip_length = 25
    args.reinit_rm = True
    return args