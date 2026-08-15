#!/bin/bash
#
# array.sbatch
#

#SBATCH --job-name=missing_real_data
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out

source /home/$USER/.bashrc
conda activate psd_general

# experiment_psd.py parameters:
#   --p               probability missing components
#   --constants_log_det
#   --inv_var
#   --percentage_out
#   --niter           number of Sinkhorn iterations
#   --nexp            number of experiments
#   --lim_nodes       max number of anchor nodes the model can bear
#   --l_rate_nodes    learning rate for the anchor nodes
#   --nbr_bounce      number of bounces between the PSD matrix and the nodes

datasets=(
    "airfoil_self_noise"
    "breast_cancer_diagnostic"
    "concrete_slump"
    "climate_model_crashes"
    "concrete_compression"
    "blood_transfusion"
    "connectionist_bench_vowel"
    "connectionist_bench_sonar"
    "planning_relax"
    "parkinsons"
    "qsar_biodegradation"
    "iris"
    "seeds"
    "ecoli"
    "glass"
    "ionosphere"
    "wine"
    "yacht_hydrodynamics"
    "yeast"
    "wine_quality_red"
)

datasets1=(
    "iris"
    "wine"
)

echo "current dataset: ${datasets1[$SLURM_ARRAY_TASK_ID]}"

python3 -u main_real_dataset.py \
    --dataset=${datasets1[$SLURM_ARRAY_TASK_ID]} \
    --p=0.4 \
    --percentage_out=0.10 \
    --nexp=10 \
    --constants_log_det="1, 0.1, 0.01, 0.001" \
    --inv_var="1.8, 1.6, 1.4" \
    --lim_nodes=65 \
    --niter=3000 \
    -mi_nbr=10 \
    --l_rate_nodes=0.001 \
    --nbr_bounce=60 \
    --lbd_constraint=0.000001 \
    --lbd_kernel=0.000001 \
    --lbd_kernel_after=0.000001 \
    --intermediate_nwt=5 \
    --hit_opt_algo=3
