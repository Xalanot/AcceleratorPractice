#!/bin/bash
#
#SBATCH --job-name=sorting_aos_vs_soa
#SBATCH --output=sorting_aos_vs_soa.txt
#
#SBATCH -w mp-capture02

srun ./sorting_aos_vs_soa