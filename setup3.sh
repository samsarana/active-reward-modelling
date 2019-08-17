#!/bin/bash
pip install -r requirements.txt
pip install .
conda install scipy=1.2.0 # required by render() function of GridworldEnv