#!/bin/bash
python main.py --env=acrobot           --n_runs=1 --n_rounds=10 --default_settings=acrobot_sam --info=RL
python main.py --env=acrobot           --n_runs=1 --n_rounds=10 --default_settings=acrobot_sam --info=RandAcq-lr0   --lr_rm=0.
python main.py --env=acrobot           --n_runs=1 --n_rounds=10 --default_settings=acrobot_sam --info=BALD-lr0      --lr_rm=0. --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --env=acrobot_scrambled --n_runs=1 --n_rounds=10 --default_settings=acrobot_sam --info=RandAcq-scram
python main.py --env=acrobot_scrambled --n_runs=1 --n_rounds=10 --default_settings=acrobot_sam --info=BALD-scram               --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5