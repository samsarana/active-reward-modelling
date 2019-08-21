#!/bin/bash
# random policy
python main.py --info=rp --env_str=gridworld --random_policy --n_rounds=5 --n_agent_steps=150000
# try normalising rewards across prefs buffer. try 3 different learning rates
python main.py --info=RL_norm-1e-4      --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --agent_gets_dones --lr_agent=1e-4
python main.py --info=RL_norm-5e-4      --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --agent_gets_dones --lr_agent=5e-4
python main.py --info=RL_norm-1e-3      --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --agent_gets_dones --lr_agent=1e-3
# try huber loss
python main.py --info=RL_huber-1e-4     --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --agent_gets_dones --lr_agent=1e-4 --no_normalise_rewards --dqn_loss=huber
# try not giving dqn done signals
python main.py --info=RL_no_dones-1e-4  --env_str=gridworld --RL_baseline --default_settings=gridworld_nb                    --lr_agent=1e-4 --no_normalise_rewards