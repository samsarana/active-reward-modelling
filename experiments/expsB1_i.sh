#!/bin/bash
python main.py --env=acrobot_hard --default_settings=acrobot_sam --n_runs=10 --n_rounds=100 --info=BALD --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --env=acrobot_hard --default_settings=acrobot_sam --n_runs=10 --n_rounds=100 --info=RL --RL_baseline