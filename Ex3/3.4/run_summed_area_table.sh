#!/bin/bash
#
#SBATCH --job-name=summed_area_table
#SBATCH --output=summed_area_table.txt
#
#SBATCH -w mp-capture01

srun ./summed_area_table