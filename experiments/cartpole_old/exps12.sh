#!/bin/bash
python main.py --info=active-BALD-3ens-forced-1 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice
python main.py --info=active-BALD-3ens-forced-2 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice
python main.py --info=active-BALD-3ens-forced-3 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice

python main.py --info=active-max_ent-3ens-forced-1 --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice
python main.py --info=active-max_ent-3ens-forced-2 --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice
python main.py --info=active-max_ent-3ens-forced-3 --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice

python main.py --info=active-naive_variance-3ens-forced-1 --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice
python main.py --info=active-naive_variance-3ens-forced-2 --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice
python main.py --info=active-naive_variance-3ens-forced-3 --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=3 --force_label_choice