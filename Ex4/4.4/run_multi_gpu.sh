#!/bin/bash
#
#SBATCH --job-name=multi_gpu
#SBATCH --output=multi_gpu.txt
#
#SBATCH -w mp-capture02

srun ./multi_gpu