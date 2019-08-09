#!/bin/bash
python main.py --env=acrobot_hard --default_settings=acrobot_sam --n_runs=10 --n_rounds=100 --info=RL-s --RL_baseline
python main.py --env=acrobot_hard --default_settings=openai --n_runs=10 --n_rounds=100 --info=RL-o --RL_baseline