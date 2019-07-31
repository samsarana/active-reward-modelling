#!/bin/bash
python main.py --info=baseline-1-reinit_rm+agent                                                                          --reinit_rm --reinit_agent --n_agent_train_steps=6000
python main.py --info=active-BALD-5ens-reinit_rm+agent --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --reinit_rm --reinit_agent --n_agent_train_steps=6000