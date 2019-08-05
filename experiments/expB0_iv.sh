#!/bin/bash
# see https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py#L95
python main.py --env=acrobot --info=RL_openai_default --RL_baseline --terminate_once_solved --n_rounds=1 --n_runs=5 \
               --lr_agent=5e-4 --n_agent_train_steps=100000 --n_agent_total_steps=100000  --replay_buffer_size=50000 \
               --epsilon_stop=0.02 --agent_gdt_step_period=1 --agent_learning_start=1000 --gamma=1.0 --target_update_period=500 --target_update_tau=1

# see https://github.com/openai/baselines/blob/master/baselines/deepq/defaults.py
python main.py --env=acrobot --info=RL_openai_atari --RL_baseline --terminate_once_solved --n_rounds=1 --n_runs=5 \
               --lr_agent=1e-4 --n_agent_train_steps=100000 --n_agent_total_steps=100000  --replay_buffer_size=10000 \
               --epsilon_stop=0.01 --agent_gdt_step_period=4 --agent_learning_start=10000 --gamma=0.99 --target_update_period=1000 --target_update_tau=1