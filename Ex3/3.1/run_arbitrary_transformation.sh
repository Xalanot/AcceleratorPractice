#!/bin/bash
#
#SBATCH --job-name=arbitrary_transformation
#SBATCH --output=arbitrary_transformation.txt
#
#SBATCH -w mp-capture01

srun ./arbitrary_transformation