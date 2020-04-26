#!/bin/bash
#
#SBATCH --job-name=weld_vertices
#SBATCH --output=weld_vertices.txt
#
#SBATCH -w mp-capture02

srun ./weld_vertices