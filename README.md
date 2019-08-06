# barm
Bayesian Active Reward Modeling.

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

If running on a VM, clone the repo into the home directory and run these commands to setup everything from scratch:
```
cd ~/barm
bash ./setup1.sh
source ~/.bashrc
bash ./setup2.sh
conda activate gym
bash ./setup3.sh
```
Also run ```bash ./setup4.sh``` if you want to be able to use the ```--save_video``` flag to save .mp4 files of agent behaviour.
