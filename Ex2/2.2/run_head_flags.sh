#!/bin/bash
#
#SBATCH --job-name=head_flags
#SBATCH --output=head_flags.txt
#
#SBATCH -w mp-capture02

srun ./head_flags