#!/bin/bash
cd ~/tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda update -n base -c defaults conda
conda create -n gym python=3.7
conda activate gym
cd barm
pip install -r requirements.txt
pip install .