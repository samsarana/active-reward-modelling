#!/bin/bash
python main.py --info=baseline-1-reinit_rm-404010 --reinit_rm
python main.py --info=active-BALD-5ens-reinit_rm-404010 --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --reinit_rm