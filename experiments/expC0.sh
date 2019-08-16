#!/bin/bash
# cartpole hyperparams for 10 rounds = 80k steps, wiht lr \in {1e-3, 5e-4, 1e-4}
python main.py --env=frozen_lake --RL_baseline --n_runs=1 --n_rounds=10 --lr_agent=1e-3 --default_settings=cartpole --info=RL_fl_cp_1e-3
python main.py --env=frozen_lake --RL_baseline --n_runs=1 --n_rounds=10 --lr_agent=5e-4 --default_settings=cartpole --info=RL_fl_cp_5e-4
python main.py --env=frozen_lake --RL_baseline --n_runs=1 --n_rounds=10 --lr_agent=1e-4 --default_settings=cartpole --info=RL_fl_cp_1e-4
# openai_default hyperparams for 1 round = 100k steps, with lr \in {1e-3, 5e-4, 1e-4}
python main.py --env=frozen_lake --RL_baseline --n_runs=1 --n_rounds=1 --lr_agent=1e-3 --default_settings=openai_defaults --info=RL_fl_opai_1e-3
python main.py --env=frozen_lake --RL_baseline --n_runs=1 --n_rounds=1 --lr_agent=5e-4 --default_settings=openai_defaults --info=RL_fl_opai_5e-4
python main.py --env=frozen_lake --RL_baseline --n_runs=1 --n_rounds=1 --lr_agent=1e-4 --default_settings=openai_defaults --info=RL_fl_opai_1e-4