#!/bin/bash
python main.py --info=baseline-1-404010
python main.py --info=active-BALD-5ens-404010 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --info=baseline-0-404010 --random_policy
python main.py --info=RL-404010 --RL_baseline