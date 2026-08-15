#!/bin/bash
#
# array.sbatch
#
# Allocated resources are NOT SHARED across the jobs.
# They represent resources allocated for each job
# in the array.

#SBATCH --job-name=missing_real_data
#SBATCH --output=%x_%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# If needed, load your environment (conda, ...)
#
# some parameter for experiment_psd.py:
# --p: probability missing components
# --costants_log_det
# --inv_var
# --percentage_out
# --niter: number iteration sinkhorn
# --nexp: number experiences
# --lim_nodes: max nbr of nodes our model can bear
# --l_rate_nodes: learning rate nodes psd model
# --nbr_bounce: number of bounce between psd matrix and nodes
source /home/$USER/.bashrc
conda activate psd_general

datasets=("airfoil_self_noise" "breast_cancer_diagnostic" "concrete_slump" "climate_model_crashes" "concrete_compression" "blood_transfusion" "connectionist_bench_vowel" "connectionist_bench_sonar" "planning_relax" "parkinsons" "qsar_biodegradation" "iris" "seeds" "ecoli" "glass" "ionosphere" "wine" "yacht_hydrodynamics" "yeast" "wine_quality_red")
echo "current dataset: ${datasets[$SLURM_ARRAY_TASK_ID]}"
python -u experiment_psd_mi.py --dataset=${datasets[$SLURM_ARRAY_TASK_ID]} --p=0.4 --percentage_out=0.10 --nexp=10 --constants_log_det="1, 0.1, 0.01, 0.001" --inv_var="2.0, 1.6, 1.2" --lim_nodes=85 --niter=3000 -mi_nbr=10 --l_rate_nodes=0.001 --nbr_bounce=30 --lbd_constraint=0.000001 --lbd_kernel=0.000001 --lbd_kernel_after=0.000001 --intermediate_nwt=5 --hit_opt_algo=3