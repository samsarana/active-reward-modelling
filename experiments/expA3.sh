#!/bin/bash
python main.py --env=cartpole_old --default_settings=cartpole --info=RandAcq-cartpole-rep --n_sample_reps=50
python main.py --env=cartpole_old --default_settings=cartpole --info=BALD-cartpole-rep --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --n_sample_reps=50