#!/bin/bash
python main.py --enrich_reward --info=baseline-1-rand_acq-rich
python main.py --enrich_reward --info=active-BALD-5ens-rich --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --enrich_reward --info=active-mean_std-5ens-rich --active_method=mean_std --uncert_method=ensemble --size_rm_ensemble=5
python main.py --enrich_reward --info=baseline-0-rand_policy-rich --random_policy
python main.py --enrich_reward --info=RL-rich --RL_baseline