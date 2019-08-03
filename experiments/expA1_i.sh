#!/bin/bash
python main.py --info=RL --RL_baseline --terminate_once_solved --reinit_rm --reinit_agent --n_agent_train_steps=8000 --n_labels_per_round=1 --n_rounds=100 --n_runs=30
python main.py --info=random_policy --random_policy --terminate_once_solved --reinit_rm --reinit_agent --n_agent_train_steps=8000 --n_labels_per_round=1 --n_rounds=100 --n_runs=5
