#!/bin/bash
#
# array.sbatch
#

#SBATCH --job-name=missing_real_data
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

source /home/$USER/.bashrc
conda activate psd_general

# main_real_dataset.py parameters:
#   --p                    probability of missing components
#   --nexp                 number of experiments
#   --niter                number of Sinkhorn iterations
#   --psd_m                number of anchor nodes
#   --psd_eta              initial precision of the Gaussians
#   --psd_alpha            parameter inside the log of the loss, log(Tr(Q.) + alpha)
#   --psd_lbd              hyperparameter of the trace regularizer
#   --psd_mu               hyperparameter of the logdet regularizer
#   --psd_lr_nodes         learning rate for the anchor nodes
#   --psd_lr_param         learning rate for the precision
#   --psd_bounce           number of alternating minimization steps
#   --psd_gradient_steps   gradient steps per bounce
#   --psd_newton_step_Q    max Newton iterations per Q update

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

dataset=${datasets1[$SLURM_ARRAY_TASK_ID]}
echo "current dataset: ${dataset}"

python3 -u main_real_dataset.py \
    --dataset=${dataset} \
    --p=0.4 \
    --nexp=2 \
    --niter=30 \
    --psd_m=85 \
    --psd_eta=1.6 \
    --psd_alpha=1e-6 \
    --psd_lbd=1e-1 \
    --psd_mu=1e-1 \
    --psd_lr_nodes=0.001 \
    --psd_lr_param=1e-4 \
    --psd_bounce=30 \
    --psd_gradient_steps=5 \
    --psd_newton_step_Q=50
