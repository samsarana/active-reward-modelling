#!/bin/bash
# python main.py --info=RL --env_str=gridworld --RL_baseline --default_settings=gridworld_zac --no_normalise_rewards --agent_gets_dones
python main.py --info=RL --env_str=gridworld --RL_baseline --default_settings=gridworld_zac --no_normalise_rewards --agent_gets_dones --path_to_agent_state_dict=./logs/0/checkpts/agent/1-29.pt --n_runs=1 --lr_agent=5e-5