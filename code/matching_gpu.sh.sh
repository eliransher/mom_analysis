#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30000M
source /home/eliransc/projects/def-dkrass/eliransc/mom_match/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/mom_analysis/code/fit_moms_diff_matchers.py