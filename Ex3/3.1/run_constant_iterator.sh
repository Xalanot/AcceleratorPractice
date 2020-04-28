#!/bin/bash
#
#SBATCH --job-name=constant_iterator
#SBATCH --output=constant_iterator.txt
#
#SBATCH -w mp-capture01

srun ./constant_iterator