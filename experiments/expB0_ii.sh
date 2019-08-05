#!/bin/bash
python main.py --env=acrobot --info=RL --RL_tau --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --target_update_tau=1e-2
python main.py --env=acrobot --info=RL --RL_hard --terminate_once_solved --n_agent_train_steps=10000 --n_labels_per_round=5 --n_rounds=100 --n_runs=2 --target_update_period=5000 --target_update_tau=1 --agent_gdt_step_period=5