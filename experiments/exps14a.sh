#!/bin/bash
python main.py --info=RL --RL_baseline
python main.py --info=baseline-0-random_policy --random_policy
python main.py --info=baseline-1-random_acq
python main.py --info=baseline-2-random_acq-5ens --size_rm_ensemble=5