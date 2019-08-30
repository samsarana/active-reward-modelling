#!/bin/bash
python main.py --info=RandAcq       --env_str=gridworld --default_settings=gridworld_zac --agent_gets_dones  --n_epochs_train_rm=30000 --grid_size=4 --grid_n_lavas 1
python main.py --info=RandAcq-5ens   --env_str=gridworld --default_settings=gridworld_zac --agent_gets_dones  --n_epochs_train_rm=30000 --grid_size=4 --grid_n_lavas 1 --size_rm_ensemble=5