#!/bin/bash
python main.py --env_str=cartpole --default_settings=cartpole --n_rounds=20 --n_runs=20 --info=RL --RL_baseline
python main.py --env_str=cartpole --default_settings=cartpole --n_rounds=20 --n_runs=20 --info=RandAcq
python main.py --env_str=cartpole --default_settings=cartpole --n_rounds=20 --n_runs=20 --info=RandAcq-5ens --size_rm_ensemble=5