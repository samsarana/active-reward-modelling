#!/bin/bash
python main.py --env=acrobot_hard --default_settings=acrobot_sam --info=BALD-s --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --env=acrobot_hard --default_settings=acrobot_sam --info=RandAcq-5ens-s --size_rm_ensemble=5