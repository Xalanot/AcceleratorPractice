#!/bin/bash
#
#SBATCH --job-name=saxpy2
#SBATCH --output=saxpy2.txt
#
#SBATCH -w mp-capture02
#SBATCH --mem-per-gpu=4G

srun ./saxpy2