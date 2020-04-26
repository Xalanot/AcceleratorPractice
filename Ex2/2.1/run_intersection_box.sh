#!/bin/bash
#
#SBATCH --job-name=inter_b
#SBATCH --output=intersection_box.txt
#
#SBATCH -w mp-capture02

srun ./intersection_box