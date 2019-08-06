#!/bin/bash
xvfb-run -a python main.py --env=acrobot --info=RL --RL_baseline --terminate_once_solved --n_agent_train_steps=10000 --n_agent_total_steps=10000 --n_rounds=10 --n_runs=5 --save_video
xvfb-run -a python main.py --env=acrobot --info=RL_gamma --RL_baseline --terminate_once_solved --n_agent_train_steps=10000 --n_agent_total_steps=10000 --n_rounds=10 --n_runs=5 --gamma=0.99 --save_video

# see https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py#L95
xvfb-run -a python main.py --env=acrobot --info=RL_openai_default_partial --RL_baseline --terminate_once_solved --n_rounds=10 --n_runs=5 \
               --lr_agent=5e-4 --n_agent_train_steps=10000 --n_agent_total_steps=10000  --replay_buffer_size=50000 \
               --epsilon_stop=0.02 --agent_gdt_step_period=1 --agent_learning_start=1000 --gamma=1.0 --target_update_period=500 --target_update_tau=1 --save_video

# see https://github.com/openai/baselines/blob/master/baselines/deepq/defaults.py
xvfb-run -a python main.py --env=acrobot --info=RL_openai_atari_partial --RL_baseline --terminate_once_solved --n_rounds=10 --n_runs=5 \
               --lr_agent=1e-4 --n_agent_train_steps=10000 --n_agent_total_steps=10000  --replay_buffer_size=10000 \
               --epsilon_stop=0.01 --agent_gdt_step_period=4 --agent_learning_start=1000 --gamma=0.99 --target_update_period=1000 --target_update_tau=1 --save_video


# see https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py#L95
xvfb-run -a python main.py --env=acrobot --info=RL_openai_default_full --RL_baseline --terminate_once_solved --n_rounds=1 --n_runs=5 \
               --lr_agent=5e-4 --n_agent_train_steps=100000 --n_agent_total_steps=100000  --replay_buffer_size=50000 \
               --epsilon_stop=0.02 --agent_gdt_step_period=1 --agent_learning_start=1000 --gamma=1.0 --target_update_period=500 --target_update_tau=1 --save_video

# see https://github.com/openai/baselines/blob/master/baselines/deepq/defaults.py
xvfb-run -a python main.py --env=acrobot --info=RL_openai_atari_full --RL_baseline --terminate_once_solved --n_rounds=1 --n_runs=5 \
               --lr_agent=1e-4 --n_agent_train_steps=100000 --n_agent_total_steps=100000  --replay_buffer_size=10000 \
               --epsilon_stop=0.01 --agent_gdt_step_period=4 --agent_learning_start=10000 --gamma=0.99 --target_update_period=1000 --target_update_tau=1 --save_video