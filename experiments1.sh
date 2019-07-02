#!/bin/sh
python main.py --info='baseline-0-RL' --RL_baseline=True --n_rounds=1 --n_epochs_pretrain_rm=0
python main.py --info='baseline-1-random'
python main.py --info='baseline-2a-ens3' --size_rm_ensemble=3
python main.py --info='baseline-2b-ens10' --size_rm_ensemble=10

python main.py --info='active-ens3' --size_rm_ensemble=3 --active_learning='ensemble_variance'
python main.py --info='active-ens10' --size_rm_ensemble=10 --active_learning='ensemble_variance'

python main.py --info='active-MC-k10-p.2' --active_learning='MC_variance' --num_MC_samples=10
python main.py --info='active-MC-k100-p.2' --active_learning='MC_variance' --num_MC_samples=100
python main.py --info='active-MC-k10-p.4' --active_learning='MC_variance' --num_MC_samples=10 --p_dropout_rm=0.4
python main.py --info='active-MC-k100-p.4' --active_learning='MC_variance' --num_MC_samples=100 --p_dropout_rm=0.4

python main.py --info='active-MC-k10-p.2' --active_learning='info_gain' --num_MC_samples=10
python main.py --info='active-MC-k100-p.2' --active_learning='info_gain' --num_MC_samples=100
python main.py --info='active-MC-k10-p.4' --active_learning='info_gain' --num_MC_samples=10 --p_dropout_rm=0.4
python main.py --info='active-MC-k100-p.4' --active_learning='info_gain' --num_MC_samples=100 --p_dropout_rm=0.4