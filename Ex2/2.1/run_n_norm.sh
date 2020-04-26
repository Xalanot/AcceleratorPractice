#!/bin/bash
#
#SBATCH --job-name=n_norm
#SBATCH --output=n_norm.txt
#
#SBATCH -w mp-capture02

srun ./n_norm