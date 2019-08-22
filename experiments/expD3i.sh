#!/bin/bash
# agent learning starts after 120k steps. so learns for 30k steps. therefore, since no reinitialisation, learns for 30k*5 = 150k steps in total, the same as in D2
python main.py --info=RandAcq_dones-finetune-1e-4 --env_str=gridworld --default_settings=gridworld_nb --agent_gets_dones --lr_agent=1e-4 --selection_factor=1 --no_reinit_agent --agent_learning_starts=120000 