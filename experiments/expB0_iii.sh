#!/bin/bash
python main.py --env=acrobot --info=RL --RL_replay --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --replay_buffer_size=1000000
python main.py --env=acrobot --info=RL --RL_gamma --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --gamma=0.99