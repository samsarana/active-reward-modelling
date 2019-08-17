#!/bin/bash
# RL baseline on gridworld with first attempt at default parameters
python main.py --info=grid_RL_test1 --env_str=gridworld --default_settings=gridworld --n_rounds=1 --n_runs=1 --RL_baseline --no_normalise_rewards