#!/bin/bash
python main.py --info=active-max_entropy-3ens-10x --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-max_entropy-3ens-100x --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3 --selection_factor=100
python main.py --info=active-max_entropy-3ens-1000x --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=3 --selection_factor=1000

python main.py --info=active-max_entropy-5ens-10x --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=5
python main.py --info=active-max_entropy-5ens-100x --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100
python main.py --info=active-max_entropy-5ens-1000x --active_method=max_entropy --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=1000

python main.py --info=active-var_ratios-3ens-10x --active_method=var_ratios --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-var_ratios-3ens-100x --active_method=var_ratios --uncert_method=ensemble --size_rm_ensemble=3 --selection_factor=100
python main.py --info=active-var_ratios-3ens-1000x --active_method=var_ratios --uncert_method=ensemble --size_rm_ensemble=3 --selection_factor=1000

python main.py --info=active-var_ratios-5ens-10x --active_method=var_ratios --uncert_method=ensemble --size_rm_ensemble=5
python main.py --info=active-var_ratios-5ens-100x --active_method=var_ratios --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=100
python main.py --info=active-var_ratios-5ens-1000x --active_method=var_ratios --uncert_method=ensemble --size_rm_ensemble=5 --selection_factor=1000
