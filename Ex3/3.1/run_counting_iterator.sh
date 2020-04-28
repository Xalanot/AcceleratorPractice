#!/bin/bash
#
#SBATCH --job-name=counting_iterator
#SBATCH --output=counting_iterator.txt
#
#SBATCH -w mp-capture01

srun ./counting_iterator