#!/bin/bash
#
#SBATCH --job-name=pinned_memory
#SBATCH --output=pinned_memory.txt
#
#SBATCH -w mp-capture01

for i in {25..30}
do
    free -g
    srun ./pinned_memory $((2**i))
done