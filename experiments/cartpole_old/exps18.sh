#!/bin/bash
python main.py --enrich_reward --info=baseline-1-rand_acq-rich --n_labels_per_round=20 --n_runs=10
python main.py --enrich_reward --info=active-BALD-5ens-rich --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --n_labels_per_round=20 --n_runs=10
python main.py --enrich_reward --info=baseline-0-rand_policy-rich --random_policy --n_runs=10
python main.py --enrich_reward --info=RL-rich --RL_baseline --n_runs=10
#python main.py --info=active-mean_std-5ens-rich --active_method=mean_std --uncert_method=ensemble --size_rm_ensemble=5 --n_labels_per_round=20 --n_runs=15