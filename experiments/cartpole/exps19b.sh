#!/bin/bash
python main.py --sequential_acq --info=active-BALD-5ens-seq-reinit_rm --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --reinit_rm
python main.py --sequential_acq --info=baseline-1-rand_acq-seq-reinit_rm --reinit_rm