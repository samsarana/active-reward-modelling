#!/bin/bash
python main.py --info=RL --env_str=gridworld --RL_baseline --default_settings=gridworld_zac --no_normalise_rewards --agent_gets_dones --lr_agent=1e-4 --no_reinit_agent