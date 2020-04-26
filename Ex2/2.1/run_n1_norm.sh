#!/bin/bash
#
#SBATCH --job-name=n1_norm
#SBATCH --output=n1_norm.txt
#
#SBATCH -w mp-capture02

srun ./n1_norm