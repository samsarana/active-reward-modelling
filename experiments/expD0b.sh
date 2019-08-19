#!/bin/bash
# python main.py --info=grid_rp     --env_str=gridworld --random_policy --n_agent_steps=10000
# python main.py --info=grid_rp_det --env_str=gridworld --random_policy --n_agent_steps=10000 --grid_deterministic_reset --grid_terminate_ep_if_done
python main.py --info=grid_RL_det_dones      --env_str=gridworld          --RL_baseline --default_settings=gridworld --grid_deterministic_reset --grid_terminate_ep_if_done --no_normalise_rewards --agent_gets_dones
python main.py --info=grid_RL_conv_det_dones --env_str=gridworld_det_term --RL_baseline --default_settings=gridworld --grid_deterministic_reset --grid_terminate_ep_if_done --no_normalise_rewards --agent_gets_dones --dqn_archi=cbnn