#!/bin/bash
python main.py --info=RL --RL --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2
python main.py --info=RL --RL_eps --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --epsilon_decay=0.9999
python main.py --info=RL --RL_tau --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --target_update_tau=1e-2
python main.py --info=RL --RL_hard --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --target_update_period=5000 --target_update_tau=1 --agent_gdt_step_period=5
python main.py --info=RL --RL_replay --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --replay_buffer_size=1000000
python main.py --info=RL --RL_gamma --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --gamma=0.99