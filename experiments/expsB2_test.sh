#!/bin/bash
python main.py --RL_baseline --default_settings=acrobot_sam --n_runs=1 --n_rounds=3 --info=RL_acrobot_test
python main.py --default_settings=acrobot_sam --n_runs=1 --n_rounds=3 --info=baseline_acrobot_test