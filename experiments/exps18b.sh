#!/bin/bash
python main.py --enrich_reward --info=baseline-1-rand_acq-rich-reinit_rm --reinit_rm
python main.py --enrich_reward --info=active-BALD-5ens-rich-reinit_rm --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --reinit_rm
python main.py --enrich_reward --info=active-mean_std-5ens-rich-reinit_rm --active_method=mean_std --uncert_method=ensemble --size_rm_ensemble=5 --reinit_rm
python main.py --enrich_reward --info=baseline-0-rand_policy-rich-reinit_rm --random_policy --reinit_rm
python main.py --enrich_reward --info=RL-rich-reinit_rm --RL_baseline --reinit_rm