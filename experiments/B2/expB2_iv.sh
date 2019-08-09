#!/bin/bash
python main.py --env=acrobot_hard --default_settings=openai --n_runs=10 --n_rounds=100 --info=RandAcq-o
python main.py --env=acrobot_hard --default_settings=openai --n_runs=10 --n_rounds=100 --info=mean_std-o --active_method=mean_std --uncert_method=ensemble --size_rm_ensemble=5