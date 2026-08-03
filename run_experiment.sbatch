#!/bin/bash
#
# run_experiment.sbatch
#

#SBATCH --job-name=psd_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

source /home/$USER/.bashrc
conda activate psd_general

python main_real_dataset.py \
    --dataset iris \
    --nexp 5 \
    --out_dir exps
