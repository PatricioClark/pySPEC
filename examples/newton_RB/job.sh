#!/bin/bash
#SBATCH -J ra1e6c
#SBATCH -o %x.out
#SBATCH -N 1
#SBATCH -n 40
#SBATCH -t 48:00:00

ml python
python3 newton_rb.py
