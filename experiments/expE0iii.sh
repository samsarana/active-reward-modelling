#!/bin/bash
python main.py --info=RL_norm_5e-4 --env_str=gridworld --RL_baseline --default_settings=gridworld_zac --agent_gets_dones --lr_agent=5e-4 --no_reinit_agent
python main.py --info=RL_norm_1e-3 --env_str=gridworld --RL_baseline --default_settings=gridworld_zac --agent_gets_dones --lr_agent=1e-3 --no_reinit_agent