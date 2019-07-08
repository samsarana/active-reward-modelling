python main.py --info=random_policy --random_policy
python main.py --info=baseline-0-RL --RL_baseline --n_rounds=2
python main.py --info=baseline-1-random

python main.py --info=active-BALD-10x-k10-p.2 --active_learning=info_gain

python main.py --info=active-BALD-20x --active_learning=info_gain --selection_factor=20
python main.py --info=active-BALD-50x --active_learning=info_gain --selection_factor=50
python main.py --info=active-BALD-100x --active_learning=info_gain --selection_factor=100

python main.py --info=active-BALD-T20 --active_learning=info_gain --num_MC_samples=20
python main.py --info=active-BALD-T50 --active_learning=info_gain --num_MC_samples=50

python main.py --info=active-BALD-p.15 --active_learning=info_gain --p_dropout_rm=0.15
python main.py --info=active-BALD-p.25 --active_learning=info_gain --p_dropout_rm=0.25
