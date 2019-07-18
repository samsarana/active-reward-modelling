#!/bin/bash
python main.py --info=active-BALD-5ens-not_pred --active_method=BALD             --uncert_method=ensemble --size_rm_ensemble=5 --no_ensemble_for_reward_pred
python main.py --info=active-max_ent-5ens-not_pred --active_method=max_entropy   --uncert_method=ensemble --size_rm_ensemble=5 --no_ensemble_for_reward_pred
python main.py --info=active-mean_std-5ens-not_pred --active_method=mean_std     --uncert_method=ensemble --size_rm_ensemble=5 --no_ensemble_for_reward_pred
python main.py --info=active-var_ratios-5ens-not_pred --active_method=var_ratios --uncert_method=ensemble --size_rm_ensemble=5 --no_ensemble_for_reward_pred