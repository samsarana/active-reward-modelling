python main.py --info=active-MI-k10-p.2 --active_learning=info_gain --num_MC_samples=10
python main.py --info=active-MI-k100-p.2 --active_learning=info_gain --num_MC_samples=100
python main.py --info=active-MI-k10-p.1 --active_learning=info_gain --num_MC_samples=10 --p_dropout_rm=0.1
python main.py --info=active-MI-k100-p.1 --active_learning=info_gain --num_MC_samples=100 --p_dropout_rm=0.1