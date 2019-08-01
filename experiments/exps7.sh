#!/bin/bash
python main.py --info=random_policy --random_policy
python main.py --info=baseline-0-RL --RL_baseline --n_rounds=2
python main.py --info=baseline-1-random

python main.py --info=active-BALD-3ens-10x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-BALD-3ens-50x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --selection_factor=50
python main.py --info=active-BALD-3ens-100x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --selection_factor=100
python main.py --info=active-BALD-3ens-1000x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --selection_factor=1000

python main.py --info=active-BALD-5ens-10x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --info=active-BALD-5ens-50x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=50
python main.py --info=active-BALD-5ens-100x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100
python main.py --info=active-BALD-5ens-1000x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=1000
