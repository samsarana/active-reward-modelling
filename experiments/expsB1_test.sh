#!/bin/bash
python main.py --default_settings=acrobot_sam --n_runs=1 --n_rounds=3 --info=RL_acrobot_test --RL_baseline
python main.py --default_settings=acrobot_sam --n_runs=2 --n_rounds=3 --info=baseline_acrobot_test
python main.py --default_settings=acrobot_sam --n_runs=1 --n_rounds=3 --info=active_acrobot_test --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5