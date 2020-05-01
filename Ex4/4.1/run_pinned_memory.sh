#!/bin/bash
#
#SBATCH --job-name=pinned_memory
#SBATCH --output=pinned_memory.txt
#
#SBATCH -w mp-capture01

srun ./pinned_memory