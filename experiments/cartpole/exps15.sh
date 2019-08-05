#!/bin/bash
python main.py --info=active-BALD-3ens --active_method=BALD             --uncert_method=ensemble --size_rm_ensemble=3 --n_labels_pretraining=10
python main.py --info=active-mean_std-3ens --active_method=mean_std     --uncert_method=ensemble --size_rm_ensemble=3 --n_labels_pretraining=10