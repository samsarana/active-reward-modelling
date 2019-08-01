#!/bin/bash
python main.py --info=random_policy --random_policy
python main.py --info=baseline-0-RL --RL_baseline

python main.py --info=baseline-1-random
python main.py --info=baseline-1-random-forced --force_label_choice

python main.py --info=active-BALD-5ens-100x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100
python main.py --info=active-BALD-5ens-100x-forced --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100 --force_label_choice

python main.py --info=active-naive_variance-5ens-100x --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100
python main.py --info=active-naive_variance-5ens-100x-forced --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100 --force_label_choice

python main.py --info=active-BALD-5ens-100x-v0 --acquisition_search_strategy=v0 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100
python main.py --info=active-BALD-5ens-100x-forced-v0 --acquisition_search_strategy=v0 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100 --force_label_choice

python main.py --info=active-naive_variance-5ens-100x-v0 --acquisition_search_strategy=v0 --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100
python main.py --info=active-naive_variance-5ens-100x-forced-v0 --acquisition_search_strategy=v0 --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100 --force_label_choice
