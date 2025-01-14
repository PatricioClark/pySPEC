#!/bin/bash
#SBATCH -J pepinn19
#SBATCH -o %x.out
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -t 48:00:00

ml python
python3 newton_kolmog.py
