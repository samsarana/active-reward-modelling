#!/bin/bash
python main.py --info=RL            --env_str=gridworld --default_settings=gridworld_zac --agent_gets_dones  --n_epochs_train_rm=30000 --grid_size=4 --grid_n_lavas 1 --RL_baseline
python main.py --info=random_policy --env_str=gridworld --default_settings=gridworld_zac --agent_gets_dones                            --grid_size=4 --grid_n_lavas 1 --random_policy
python main.py --info=random_rp     --env_str=gridworld --default_settings=gridworld_zac --agent_gets_dones                            --grid_size=4 --grid_n_lavas 1 --n_epochs_train_rm=0