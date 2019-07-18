#!/bin/bash
conda update -n base -c defaults conda
conda create -n gym python=3.7
conda activate gym
cd ~/barm
pip install -r requirements.txt
pip install .