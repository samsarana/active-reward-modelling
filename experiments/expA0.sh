#!/bin/bash
time xvfb-run -s "-screen 0 1400x900x24" python main.py --info=baseline --n_runs=1 --n_rounds=3 --save_video 2>&1 | tee baseline.txt
time xvfb-run -s "-screen 0 1400x900x24" python main.py --info=baseline_no_norm --n_runs=1 --n_rounds=3 --save_video --no_normalise_rm_while_training  2>&1 | tee baseline_no_norm.txt
time xvfb-run -s "-screen 0 1400x900x24" python main.py --info=baseline-reinit_rm_agent --n_runs=1 --n_rounds=3 --save_video --reinit_rm --reinit_agent  2>&1 | tee baseline-reinit_rm_agent.txt
time xvfb-run -s "-screen 0 1400x900x24" python main.py --info=active-reinit_rm_agent --active_method=BALD --uncert_method=ensemble --n_runs=1 --n_rounds=3 --save_video --reinit_rm --reinit_agent  2>&1 | tee active-reinit_rm_agent.txt