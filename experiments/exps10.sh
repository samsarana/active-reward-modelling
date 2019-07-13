#!/bin/bash
python main.py --info=random_policy --random_policy
python main.py --info=baseline-0-RL --RL_baseline

python main.py --info=baseline-1-random1
python main.py --info=baseline-1-random2
python main.py --info=baseline-1-random3
python main.py --info=baseline-1-random4
python main.py --info=baseline-1-random5

python main.py --info=baseline-1-random-forced1 --force_label_choice
python main.py --info=baseline-1-random-forced2 --force_label_choice
python main.py --info=baseline-1-random-forced3 --force_label_choice
python main.py --info=baseline-1-random-forced4 --force_label_choice
python main.py --info=baseline-1-random-forced5 --force_label_choice
