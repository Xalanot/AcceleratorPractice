#!/bin/bash
#
#SBATCH --job-name=remove_points2d
#SBATCH --output=remove_points2d.txt
#
#SBATCH -w mp-capture01

srun ./remove_points2d