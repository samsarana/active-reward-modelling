#!/bin/bash
python main.py --info=RandAcq_dones-finetune-5e-4 --env_str=gridworld --default_settings=gridworld_nb --agent_gets_dones --lr_agent=5e-4 --selection_factor=1 --no_reinit_agent --agent_learning_starts=120000