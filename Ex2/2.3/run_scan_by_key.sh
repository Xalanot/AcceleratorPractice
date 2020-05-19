#!/bin/bash
#
#SBATCH --job-name=scan_by_key
#SBATCH --output=scan_by_key.txt
#
#SBATCH -w mp-capture02

srun ./scan_by_key 1000