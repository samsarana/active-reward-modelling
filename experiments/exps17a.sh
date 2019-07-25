#!/bin/bash
python main.py --info=baseline-1-reinit --reinit_agent
python main.py --info=active-BALD-5ens-reinit --reinit_agent --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5