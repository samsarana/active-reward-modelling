#!/bin/bash
python main.py --sequential_acq --info=active-BALD-5ens-seq --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --n_labels_per_round=20 --n_runs=6 --n_epochs_train_rm=500
python main.py --sequential_acq --info=baseline-1-rand_acq-seq --n_labels_per_round=20 --n_runs=6 --n_epochs_train_rm=500