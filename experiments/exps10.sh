#!/bin/bash
python main.py --info=random_policy --random_policy
python main.py --info=baseline-0-RL --RL_baseline

python main.py --info=baseline-1-random-1
python main.py --info=baseline-1-random-2
python main.py --info=baseline-1-random-3
python main.py --info=baseline-1-random-4
python main.py --info=baseline-1-random-5

python main.py --info=baseline-1-random-forced-1 --force_label_choice
python main.py --info=baseline-1-random-forced-2 --force_label_choice
python main.py --info=baseline-1-random-forced-3 --force_label_choice
python main.py --info=baseline-1-random-forced-4 --force_label_choice
python main.py --info=baseline-1-random-forced-5 --force_label_choice
