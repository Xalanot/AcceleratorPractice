#!/bin/bash
#
#SBATCH --job-name=transform_iterator
#SBATCH --output=transform_iterator.txt
#
#SBATCH -w mp-capture01

srun ./transform_iterator