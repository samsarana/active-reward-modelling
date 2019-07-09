python main.py --info=random_policy --random_policy
python main.py --info=baseline-0-RL --RL_baseline --n_rounds=2
python main.py --info=baseline-1-random

python main.py --info=active-BALD-3ens-10x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=3
python main.py --info=active-BALD-5ens-10x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5
python main.py --info=active-BALD-10ens-10x --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=10

python main.py --info=active-BALD-MC-10x-T10-p.2 --active_method=BALD --uncert_method=MC

python main.py --info=active-BALD-MC-20x --active_method=BALD --uncert_method=MC --selection_factor=20
python main.py --info=active-BALD-MC-50x --active_method=BALD --uncert_method=MC --selection_factor=50
python main.py --info=active-BALD-MC-100x --active_method=BALD --uncert_method=MC --selection_factor=100
python main.py --info=active-BALD-MC-1000x --active_method=BALD --uncert_method=MC --selection_factor=1000

python main.py --info=active-BALD-MC-T20 --active_method=BALD --uncert_method=MC --num_MC_samples=20
python main.py --info=active-BALD-MC-T50 --active_method=BALD --uncert_method=MC --num_MC_samples=50
python main.py --info=active-BALD-MC-T100 --active_method=BALD --uncert_method=MC --num_MC_samples=100

python main.py --info=active-BALD-MC-p.15 --active_method=BALD --uncert_method=MC --p_dropout_rm=0.15
python main.py --info=active-BALD-MC-p.25 --active_method=BALD --uncert_method=MC --p_dropout_rm=0.25
