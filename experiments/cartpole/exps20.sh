#!/bin/bash
# also must run Christiano since I haven't benchmarked its performance on acquiring 20 labels per round yet
python main.py --acq_search_strategy=christiano --info=active-BALD-5ens-christiano --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --n_labels_per_round=20 --n_runs=10
python main.py --acq_search_strategy=all_pairs --info=active-BALD-5ens-all_pairs --active_method=BALD --uncert_method=ensemble --size_rm_ensemble=5 --n_labels_per_round=20 --n_runs=10
python main.py --acq_search_strategy=all_pairs --info=baseline-1-rand_acq-all_pairs --n_labels_per_round=20 --n_runs=10