#!/bin/bash
#
#SBATCH --job-name=sorting_aos_vs_soa
#SBATCH --output=sorting_aos_vs_soa.txt
#
#SBATCH -w mp-capture01

srun ./sorting_aos_vs_soa