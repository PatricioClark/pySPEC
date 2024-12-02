#!/bin/bash
#SBATCH -J newtkol4
#SBATCH -o %x.out
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -t 48:00:00
#SBATCH --nodelist=c2

ml python
python3 newton_kolmog.py
