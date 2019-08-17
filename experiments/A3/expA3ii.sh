#!/bin/bash
python main.py --env_str=cartpole --default_settings=cartpole --n_rounds=20 --n_runs=20 --info=BALD-5ens --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --env_str=cartpole --default_settings=cartpole --n_rounds=20 --n_runs=20 --info=mean_std-5ens --active_method=mean_std --uncert_method=ensemble --size_rm_ensemble=5
python main.py --env_str=cartpole --default_settings=cartpole --n_rounds=20 --n_runs=20 --info=BALD-MC.2 --p_dropout_rm=0.2  --active_method=BALD --uncert_method=MC