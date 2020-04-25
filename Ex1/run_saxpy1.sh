#!/bin/bash
#
#SBATCH --job-name=saxpy1
#SBATCH --output=saxpy1.csv
#
#SBATCH -w mp-capture02

srun ./a.out 10