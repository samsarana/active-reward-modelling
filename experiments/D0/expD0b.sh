#!/bin/bash
python main.py --info=grid_rp     --env_str=gridworld --random_policy --n_agent_steps=500000
python main.py --info=grid_rp_det --env_str=gridworld --random_policy --n_agent_steps=500000 --grid_deterministic_reset --grid_terminate_ep_if_done
python main.py --info=grid_RL_cnn_det_term_dones-1e-4  --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --no_normalise_rewards --agent_gets_dones --dqn_archi=cnn --grid_deterministic_reset --grid_terminate_ep_if_done --lr_agent=1e-4
python main.py --info=grid_RL_cnn_det_term_dones-5e-4  --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --no_normalise_rewards --agent_gets_dones --dqn_archi=cnn --grid_deterministic_reset --grid_terminate_ep_if_done --lr_agent=5e-4
python main.py --info=grid_RL_cnn_det_term_dones-1e-3  --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --no_normalise_rewards --agent_gets_dones --dqn_archi=cnn --grid_deterministic_reset --grid_terminate_ep_if_done --lr_agent=1e-3
# python main.py --info=grid_RL_cnn_term_dones    --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --no_normalise_rewards --agent_gets_dones --dqn_archi=cnn --grid_terminate_ep_if_done
# python main.py --info=grid_RL_cnn_dones         --env_str=gridworld --RL_baseline --default_settings=gridworld_nb --no_normalise_rewards --agent_gets_dones --dqn_archi=cnn