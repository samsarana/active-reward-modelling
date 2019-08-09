#!/bin/bash
python main.py --default_settings=cartpole --info=RL --RL_baseline
python main.py --default_settings=cartpole --info=baseline-1-reinit_rm+agent
python main.py --default_settings=cartpole --info=active-BALD-5ens-reinit_rm+agent --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --default_settings=cartpole --info=active-mean_std-5ens-reinit_rm+agent --active_method=mean_std --uncert_method=ensemble --size_rm_ensemble=5
python main.py --default_settings=cartpole --info=baseline-2-5ens-reinit_rm+agent --size_rm_ensemble=5
python main.py --default_settings=cartpole --info=active-BALD-MC.15-reinit_rm+agent --p_dropout_rm=0.15  --active_method=BALD --uncert_method=MC
python main.py --default_settings=cartpole --info=active-BALD-MC.2-reinit_rm+agent --p_dropout_rm=0.2 --active_method=BALD --uncert_method=MC
python main.py --default_settings=cartpole --info=active-BALD-MC.25-reinit_rm+agent --p_dropout_rm=0.25 --active_method=BALD --uncert_method=MC