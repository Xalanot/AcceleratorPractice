#!/bin/bash
#
#SBATCH --job-name=dot_product
#SBATCH --output=dot_product.txt
#
#SBATCH -w mp-capture01

srun ./dot_product