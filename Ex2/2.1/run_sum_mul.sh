#!/bin/bash
#
#SBATCH --job-name=sum_mul
#SBATCH --output=sum_mul.txt
#
#SBATCH -w mp-capture02

srun ./sum_mul