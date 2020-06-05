#!/bin/bash
#
#SBATCH --job-name=pinned_memory
#SBATCH --output=pinned_memory.txt
#
#SBATCH -w mp-capture01

for i in {28..30}
do
    srun cuda-memcheck ./pinned_memory $((2**i))
done