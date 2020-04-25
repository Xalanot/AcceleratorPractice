#!/bin/bash
#
#SBATCH --job-name=saxpy3
#SBATCH --output=saxpy3.txt
#
#SBATCH -w mp-capture02

srun ./saxpy3