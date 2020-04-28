#!/bin/bash
#
#SBATCH --job-name=sum_rows
#SBATCH --output=sum_rows.txt
#
#SBATCH -w mp-capture01

srun ./arbitrary_transformation