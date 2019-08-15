#!/bin/bash
# This experiment will establish whether frozen lake can be solved too without making any reward model updates
# NB I haven't tested how the RL baseline learns with these DQN hyperparams
python main.py --env=frozen_lake --default_settings=acrobot_sam --info=frozen_lake-lr0 --n_runs=1 --n_rounds=10 --lr_rm=0.
# --reinit_rm_when_q_learning