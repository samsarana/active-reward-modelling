#!/bin/bash
python main.py --env=acrobot_hard --default_settings=openai --info=BALD-o --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --env=acrobot_hard --default_settings=openai --info=RandAcq-5ens-o --size_rm_ensemble=5