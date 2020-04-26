#!/bin/bash
#
#SBATCH --job-name=concat
#SBATCH --output=concat.txt
#
#SBATCH -w mp-capture02

srun ./concat_vectors