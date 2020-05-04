#!/bin/bash
#
#SBATCH --job-name=simple_moving_average
#SBATCH --output=simple_moving_average.txt
#
#SBATCH -w mp-capture02

#srun nvprof --print-gpu-trace ./multi_gpu
srun ./simple_moving_average