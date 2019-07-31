#!/bin/bash
python main.py --info=baseline-1-seq-reinit_rm+agent                                                                          --sequential_acq --reinit_rm --reinit_agent
python main.py --info=active-BALD-5ens-seq-reinit_rm+agent --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --sequential_acq --reinit_rm --reinit_agent