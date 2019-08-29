#!/bin/bash
python main.py --info=BALD-no-batch --env_str=gridworld --default_settings=gridworld_zac --agent_gets_dones  --n_epochs_train_rm=30000 --grid_size=4 --grid_deterministic_reset --size_rm_ensemble=5 --uncert_method=ensemble --active_method=BALD --batch_size_acq 1 1 1 1 1 1 1 1 1 1
