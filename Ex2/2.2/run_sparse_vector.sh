#!/bin/bash
#
#SBATCH --job-name=sparse
#SBATCH --output=sparse.txt
#
#SBATCH -w mp-capture02

srun ./sparse_vector