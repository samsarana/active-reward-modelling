#!/bin/bash

python main.py --info=baseline-1-random-1 --clip_length=1
python main.py --info=baseline-1-random-2 --clip_length=2
python main.py --info=baseline-1-random-5 --clip_length=5
python main.py --info=baseline-1-random-10 --clip_length=10
python main.py --info=baseline-1-random-20 --clip_length=20
python main.py --info=baseline-1-random-25 --clip_length=25
python main.py --info=baseline-1-random-50 --clip_length=50
python main.py --info=baseline-1-random-100 --clip_length=100
python main.py --info=baseline-1-random-200 --clip_length=200
python main.py --info=baseline-1-random-500 --clip_length=500
python main.py --info=baseline-1-random-1000 --clip_length=1000

python main.py --info=active-BALD-3ens-len1 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=1
python main.py --info=active-BALD-3ens-len2 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=2
python main.py --info=active-BALD-3ens-len5 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=5
python main.py --info=active-BALD-3ens-len10 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=10
python main.py --info=active-BALD-3ens-len20 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=20
python main.py --info=active-BALD-3ens-len25 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=25
python main.py --info=active-BALD-3ens-len50 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=50
python main.py --info=active-BALD-3ens-len100 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=100
python main.py --info=active-BALD-3ens-len200 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=200
python main.py --info=active-BALD-3ens-len500 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=500
python main.py --info=active-BALD-3ens-len1000 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3 --clip_length=1000
