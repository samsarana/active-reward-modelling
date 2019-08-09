#!/bin/bash
python main.py --env=acrobot_hard --default_settings=acrobot_sam --n_runs=10 --n_rounds=100 --info=RandAcq-s
python main.py --env=acrobot_hard --default_settings=acrobot_sam --n_runs=10 --n_rounds=100 --info=mean_std-s --active_method=mean_std --uncert_method=ensemble --size_rm_ensemble=5