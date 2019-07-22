#!/bin/bash
python main.py --info=baseline-1-5L --n_labels_per_round=5
python main.py --info=baseline-1-10L --n_labels_per_round=10
python main.py --info=baseline-1-20L --n_labels_per_round=20
python main.py --info=baseline-1-30L --n_labels_per_round=30
python main.py --info=baseline-1-40L --n_labels_per_round=40
python main.py --info=baseline-1-50L --n_labels_per_round=50
python main.py --info=baseline-1-75L --n_labels_per_round=75
python main.py --info=baseline-1-100L --n_labels_per_round=100