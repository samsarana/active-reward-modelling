# barm
Bayesian Active Reward Modeling

Make a fresh conda env with Python 3.7, then: 
```
pip install -r requirements.txt
pip install .
```
Now you can do something like:
```
python main.py
```
to do RL from synthetic preferences in CartPole using random acquisition, or something like:
```
python main.py --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
```
to use BALD acquisition function.

If running on a VM, run these commands to setup everything from scratch:
bash ./barm/setup1.sh
source ~/.bashrc
bash ./barm/setup2.sh
conda activate gym
bash ./barm/setup3.sh
