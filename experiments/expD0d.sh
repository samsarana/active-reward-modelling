#!/bin/bash
python main.py --info=grid_RL_cnn_dones-1e-4           --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --no_normalise_rewards --agent_gets_dones --dqn_archi=cnn --lr_agent=1e-4
python main.py --info=grid_RL_cnn_dones-5e-4           --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --no_normalise_rewards --agent_gets_dones --dqn_archi=cnn --lr_agent=5e-4
python main.py --info=grid_RL_cnn_dones-1e-3           --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --no_normalise_rewards --agent_gets_dones --dqn_archi=cnn --lr_agent=1e-3