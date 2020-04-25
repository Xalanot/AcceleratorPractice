#!/bin/bash
#
#SBATCH --job-name=saxpy1
#SBATCH --output=saxpy1.txt
#
#SBATCH -w mp-capture02

srun ./saxpy2