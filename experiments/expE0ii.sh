#!/bin/bash
python main.py --info=RL_norm_1e-4 --env_str=gridworld --RL_baseline --default_settings=gridworld_zac --agent_gets_dones --lr_agent=1e-4 --no_reinit_agent