#!/bin/bash
# D1a
python main.py --info=RandAcq_dones-1e-4           --env_str=gridworld --default_settings=gridworld_nb --agent_gets_dones --lr_agent=1e-4 --p_dropout_rm=0.5 --n_labels_per_round=200 --batch_size_acq=200 \
 --n_rounds=5 --n_agent_steps=150000 --rm_archi=cnn --exploration_fraction=0.33 --agent_test_frequency=15