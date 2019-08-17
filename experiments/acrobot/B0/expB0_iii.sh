#!/bin/bash
python main.py --env=acrobot --info=RL_replay --RL_baseline --terminate_once_solved --n_agent_train_steps=10000 --n_agent_total_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=3 --replay_buffer_size=1000000
python main.py --env=acrobot --info=RL_gamma --RL_baseline --terminate_once_solved --n_agent_train_steps=10000 --n_agent_total_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=3 --gamma=0.99