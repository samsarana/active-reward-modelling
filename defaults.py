def cartpole_defaults(args):
    """NB to truly recover my old defaults,
       should set n_agent_steps := 30000
       and agent_learning_start := 27000
       Even then, there's still a slight difference
       because the extra steps are now
       taken at beginning of training. But I think
       this isn't an important moving part.
       Also I changed the episilon annealing
       from exponential to linear
       but I think it should be roughly the same
    """
    # args.lr_agent = 1e-3
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
    # args.n_rounds = 50
    # args.n_runs = 20
    return args


def acrobot_sam_defaults(args):
    # args.lr_agent = 1e-3
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
    args.n_epochs_pretrain_rm = 1500 # not yet tested
    args.n_epochs_train_rm = 1500 # not yet tested
    args.clip_length = 25
    args.reinit_rm = True
    return args


def openai_defaults(args):
    """NB These settings succeeded for Acrobot
       though not quite as well as acrobot_sam_defaults
    """
    # args.lr_agent = 5e-4
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
    # args.lr_agent = 1e-4
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


def gridworld_zac_defaults(args):
    """
    Hyperparams given on p.4 of:
    https://arxiv.org/pdf/1907.01475.pdf
    """
    args.h1_agent = 256
    args.h2_agent = 256
    args.h3_agent = 512
    args.batch_size_agent = 32
    args.optimizer_agent = 'RMSProp'
    args.lambda_agent = 0
    args.lr_agent = 1e-4
    args.replay_buffer_size = int(10e3)
    args.target_update_period = 1000
    args.target_update_tau = 1 # I'm guessing they used hard updates
    # zac uses exponential annealing but this scheme is roughly equivalent
    # log_0.999(0.05) = 2994 ~ 3000
    # learning update 3000 = 9k agent steps (assuming learning update every 3 steps)
    # train for 3M RL steps
    # so exploration fraction is 9000/3e6 = 0.003. This doesn't seem like enough, let's use 0.1
    args.episilon_start = 1.0
    args.epsilon_stop = 0.05
    args.exploration_fraction = 0.1
    # from here on, they don't mention their values
    # args.agent_gdt_step_period = 4 # Zac makes gradient updates at the end of each episode. My code now does the same
    args.gamma = 0.99 # this is standard
    args.n_agent_steps = int(3e6) # may need to increase since Zac trains for 1M *episodes*, so a lot more
    args.n_agent_steps_pretrain = 0 # not sure if Zac does pretraining
    args.agent_test_frequency = 30 # test every 100k agent steps
    # reward modelling
    args.n_epochs_pretrain_rm = 2000 # not yet tested
    args.n_epochs_train_rm = 2000 # not yet tested
    return args


def gridworld_nb_defaults(args):
    """
    Sets hyerparams to be those given in:
    https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
    """
    # These values I got from the notebook
    args.dqn_archi = 'cnn'
    args.batch_size_agent = 32
    args.agent_gdt_step_period = 4 
    args.gamma = 0.99
    args.episilon_start = 1.0
    args.epsilon_stop = 0.1
    args.exploration_fraction = 0.33 # annealing_steps / n_agent_steps = 10k / 500k = 0.02. NB I increased this after first (failed) experiment
    # TODO I'm unsure whether having agent learning starting after exploration has "finished" is a problem (expD3)
    # args.n_agent_steps = int(500e3) # num_episodes * max_epLen = 10k * 50 = 500k
    # args.n_agent_steps_pretrain = 10000
    args.target_update_tau = 0.001
    args.target_update_period = 4
    args.replay_buffer_size = int(50e3)
    # args.agent_test_frequency = 15 # test every 10k agent steps (we take 150k steps each round)
    args.rm_archi = 'cnn_mod'
    # args.n_epochs_train_rm = 3000 # not yet sure if this will train to convergence... check plots!
    return args