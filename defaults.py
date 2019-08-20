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
    Used a combination of hyerparams given in:
    https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
    and used by Zac in the function dqn()
    I've commented which ones I used.
    In hindsight, I think it was silly to use a combination.
    The model archi used by Zac and notebook are very different
    and since I used Zac's archi, I think I should have just copied
    Zac's hyperparams exactly.
    NB Zac uses RMSProp as optimizer.
    Notebook uses Adam, which I also used for the first attempt
    """
    # These values I got from the notebook
    args.batch_size_agent = 32 # Zac uses 64
    args.agent_gdt_step_period = 4 # I couldn't find the value Zac uses
    args.gamma = 0.99 # Zac and notebook
    args.episilon_start = 1.0 # Zac and notebook
    args.epsilon_stop = 0.1 # Zac uses 0.05
    args.exploration_fraction = 0.1 # annealing_steps / n_agent_steps = 10k / 500k = 0.02. NB I increased this after first (failed) experiment
    # NB Zac does exponential annealing & I haven't checked how these compare
    args.n_agent_steps = int(500e3) # num_episodes * max_epLen = 10k * 50 = 500k. Zac uses 100k
    args.n_agent_steps_pretrain = 10000 # not sure if Zac does pretraining
    args.target_update_tau = 0.001 # Big difference here: Zac uses hard updates (I guess?)
    args.target_update_period = 4 # Zac uses 5. but wait, in their paper, they use 1K...
    # from here on, I got the values from Zac's defaults
    args.h1_agent = 128 # used by Zac. noteboook has 4 conv layers followed by size 512 layer,
    args.h2_agent = 128 # then "split into separate advantage and value streams." (they use dueling DQN)
    args.replay_buffer_size = int(20e3) # used by Zac. NB noteboook uses 50e3
    args.lr_agent = 1e-3 # used by Zac. NB noteboook uses 1e-4 but I'll start w Zac's since my archi is same as his, not theirs
    args.agent_test_frequency = 50 # test every 10k agent steps (50 times in total)
    # TODO think about these reward modelling settings
    args.n_epochs_pretrain_rm = 2000 # not yet tested
    args.n_epochs_train_rm = 2000 # not yet tested
    return args


def gridworld_nb_defaults(args):
    """
    Used a combination of hyerparams given in:
    https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
    and used by Zac in the function dqn()
    I've commented which ones I used.
    In hindsight, I think it was silly to use a combination.
    The model archi used by Zac and notebook are very different
    and since I used Zac's archi, I think I should have just copied
    Zac's hyperparams exactly.
    NB Zac uses RMSProp as optimizer.
    Notebook uses Adam, which I also used for the first attempt
    """
    # These values I got from the notebook
    args.dqn_archi = 'cnn'
    args.batch_size_agent = 32
    args.agent_gdt_step_period = 4 
    args.gamma = 0.99
    args.episilon_start = 1.0
    args.epsilon_stop = 0.1
    args.exploration_fraction = 0.1 # annealing_steps / n_agent_steps = 10k / 500k = 0.02. NB I increased this after first (failed) experiment
    args.n_agent_steps = int(500e3) # num_episodes * max_epLen = 10k * 50 = 500k. Zac uses 100k
    args.n_agent_steps_pretrain = 10000
    args.target_update_tau = 0.001
    args.target_update_period = 4
    # from here on, I got the values from Zac's defaults
    args.replay_buffer_size = int(50e3)
    args.agent_test_frequency = 50 # test every 10k agent steps (50 times in total)
    # TODO think about these reward modelling settings
    args.rm_archi = 'cnn'
    args.n_epochs_pretrain_rm = 2000 # not yet tested
    args.n_epochs_train_rm = 2000 # not yet tested
    return args