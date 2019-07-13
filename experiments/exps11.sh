#!/bin/bash
python main.py --info=active-BALD-3ens-1 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-BALD-3ens-2 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-BALD-3ens-3 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3

python main.py --info=active-max_ent-3ens-1 --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-max_ent-3ens-2 --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-max_ent-3ens-3 --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3

python main.py --info=active-naive_variance-3ens-1 --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-naive_variance-3ens-2 --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-naive_variance-3ens-3 --active_method=naive_variance --uncert_method=ensemble --size_rm_ensemble=3