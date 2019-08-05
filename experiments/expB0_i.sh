#!/bin/bash
python main.py --env=acrobot --info=RL --RL --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2
python main.py --env=acrobot --info=RL --RL_eps --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --epsilon_decay=0.9999