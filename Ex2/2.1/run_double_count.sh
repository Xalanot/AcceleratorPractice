#!/bin/bash
#
#SBATCH --job-name=double_count
#SBATCH --output=double_count.txt
#
#SBATCH -w mp-capture02

srun ./double_count