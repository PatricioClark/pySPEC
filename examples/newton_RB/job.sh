#!/bin/bash
#SBATCH -J ra1e6
#SBATCH -o %x.out
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 48:00:00

ml python
python3 newton_rb.py
