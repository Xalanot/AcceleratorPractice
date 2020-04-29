#!/bin/bash
#
#SBATCH --job-name=padded_grid_reduction
#SBATCH --output=padded_grid_reduction.txt
#
#SBATCH -w mp-capture01

srun ./padded_grid_reduction