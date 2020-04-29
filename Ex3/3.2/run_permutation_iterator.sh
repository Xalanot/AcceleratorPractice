#!/bin/bash
#
#SBATCH --job-name=permutation_iterator
#SBATCH --output=permutation_iterator.txt
#
#SBATCH -w mp-capture01

srun ./permutation_iterator