#!/bin/bash
#
# array.sbatch
#

#SBATCH --job-name=missing_real_data
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out

source /home/$USER/.bashrc
conda activate psd_general

# main_real_dataset.py parameters:
#   --p                       probability of missing components
#   --nexp                    number of experiments
#   --perc_test_set           percentage of dataset held out as a test set (datasets at or below
#                             --large_dataset_threshold only)
#   --large_dataset_threshold datasets with more observations than this use --max_train_obs for
#                             training and put everything else in the test set, ignoring
#                             --perc_test_set
#   --max_train_obs           training-set size used for datasets over --large_dataset_threshold
#   --sinkhorn_niter          number of Sinkhorn iterations
#   --psd_num_anchor_nodes    number of anchor nodes
#   --psd_anchor_init         data_subset: anchor nodes are a random subset of the initially-
#                             imputed data. uniform_hypercube: anchor nodes are sampled i.i.d.
#                             uniformly in the bounding box of that same data instead
#   --psd_anchor_impute       how missing entries are filled to place the anchor nodes, in both the
#                             cross-validation fits and the real fit: mean (default), median,
#                             most_frequent, or constant (fills with 0). Only positions the anchor
#                             nodes -- never produces the reported imputation
#   --psd_eta                 initial LOG-precision of the Gaussians (actual starting precision = exp(this))
#   --psd_eta_init_mode       fixed: every dimension starts at --psd_eta. empirical: every
#                             dimension starts at 1 / empirical_variance instead, ignoring --psd_eta
#   --psd_alpha               parameter inside the log of the loss, log(Tr(Q.) + alpha)
#   --psd_lbd                 hyperparameter of the trace regularizer
#   --psd_mu                  hyperparameter of the logdet regularizer
#   --psd_lr_nodes            learning rate for the anchor nodes
#   --psd_lr_param            learning rate for the precision
#   --psd_bounce              number of alternating minimization steps (1 = Newton solve only,
#                             no gradient steps on anchor nodes/precision)
#   --psd_gradient_steps      gradient steps per bounce
#   --psd_newton_step_Q       max Newton iterations per Q update
#   --verbose_newton          print every Newton iteration (loss, decrement) inside every bounce's
#                             inner Q update, separately from --verbose's per-5-bounce summary
# THIS FILE RUNS THE DISTRIBUTIONAL CV (--psd_distributional_cross_validate). That flag overrides
# every other CV mode, so --psd_cross_validate / --psd_anchor_cross_validate / --psd_cv_training_rows
# / --psd_cv_num_anchor_nodes / --psd_cv_num_bounces / --psd_cv_metric are deliberately NOT passed
# below -- they would be read by nobody. To switch modes, swap the one flag and put them back.
#
#   --psd_cv_lr_nodes_grid    comma-separated candidate values for --psd_lr_nodes
#   --psd_cv_lr_param_grid    comma-separated candidate values for --psd_lr_param
#   --psd_cv_eta_grid         comma-separated candidate LOG-precision starting points for --psd_eta
#   --psd_cv_metric           validation metric for --psd_cross_validate: rmse (default), ed, or
#                             ot. RMSE/ED/OT are always all computed and printed for every
#                             candidate; this only picks which one chooses the winner
#   --psd_anchor_cross_validate  after hyperparameters are chosen, split the training rows into a
#                             training and a validation part, try several random anchor-node
#                             subsamples of the training part, score each by training a
#                             Newton-only fit on the training part and imputing the held-out
#                             validation part, and use the best one
#   --psd_cv_anchor_candidates_per_split  each split, sample this number of set of anchor nodes. Each set contains psd_number_anchor_nodes
#   --psd_cv_anchor_num_splits  independent training/validation splits to draw (1 = a single
#                             split, i.e. every candidate sits one common exam). Raising it stops
#                             any row being permanently excluded from being anchor material and
#                             makes the winner less split-dependent, at proportional cost
#   --psd_cv_anchor_training_rows  rows each candidate is TRAINED on, per split -- the knob that
#                             sets the cost, since the Newton solve scales with the rows fit on
#   --psd_cv_anchor_validation_rows  rows each candidate is SCORED on, per split (unset = every
#                             row left over after the training rows, which is free variance
#                             reduction since scoring is far cheaper than fitting)
#   --psd_cv_anchor_newton_iterations  max Newton iterations per anchor-node candidate
#   --psd_cv_anchor_metric    validation metric for --psd_anchor_cross_validate: rmse (default),
#                             ed, or ot. ed/ot compare whole rows, so entries already missing
#                             before extra-hiding are zeroed out in both the imputed and true
#                             arrays first, so the score only reflects recovery of the
#                             extra-hidden entries
#   --psd_cv_joint_num_bounces  bounces per fit for --psd_joint_cross_validate (small on purpose:
#                             the joint search runs |grid| x candidates FULL fits, so cost scales
#                             directly with this)
#   --psd_joint_cross_validate  search the hyperparameters AND the anchor nodes JOINTLY as one
#                             grid, instead of the greedy --psd_cross_validate then
#                             --psd_anchor_cross_validate sequence (which freezes the
#                             hyperparameters before choosing anchors, so it cannot see any
#                             interaction between them). Overrides both. Costs
#                             splits x |grid| x candidates fits -- shrink the grids first
#   --psd_distributional_cross_validate  same joint search, but scored DISTRIBUTIONALLY: the extra
#                             holes go into the TRAINING rows, the held-out rows are left untouched
#                             as a reference sample of real data, and each candidate is scored on
#                             how closely its imputed training cloud resembles that reference
#                             cloud. Asks "does the filled-in data look real" rather than "did we
#                             recover these values". It is also the only CV that imputes the SAME
#                             rows it fit on, matching what the real run does. Overrides every
#                             other CV flag; --psd_cv_anchor_metric must be ed or ot (not rmse).
#                             Same cost as the joint search
#   --psd_verbose_metrics     alongside --verbose, also print ORACLE MAE/RMSE/OT/ED every 5th
#                             bounce (and the last bounce): mean-imputed dataset vs. the true
#                             underlying training data, at the entries hidden from the fit --
#                             oracle since the fit itself never sees those true values. More
#                             expensive, opt-in separately from the rest of --verbose's output

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
     "airfoil_self_noise"
     "yeast"
     "blood_transfusion"
     "concrete_compression"   
)

dataset=${datasets1[$SLURM_ARRAY_TASK_ID]}
echo "current dataset: ${dataset1}"

python3 -u main_real_dataset.py \
    --dataset=${dataset} \
    --p=0.2 \
    --nexp=5 \
    --perc_test_set=0.3 \
    --large_dataset_threshold=1000 \
    --max_train_obs=800 \
    --sinkhorn_niter=3000 \
    --psd_num_anchor_nodes=65 \
    --psd_anchor_impute=constant \
    --psd_eta=1.6 \
    --psd_alpha=1e-6 \
    --psd_lbd=1e-1 \
    --psd_mu=1e-3 \
    --psd_lr_nodes=1e-3 \
    --psd_lr_param=1e-4 \
    --psd_bounce=5 \
    --psd_gradient_steps=2 \
    --psd_newton_step_Q=3 \
    --psd_cv_lr_nodes_grid=3e-4,1e-3,3e-3 \
    --psd_cv_lr_param_grid=3e-5,1e-4,3e-4,1e-3 \
    --psd_cv_eta_grid=-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0 \
    --psd_distributional_cross_validate \
    --psd_cv_joint_num_bounces=5 \
    --psd_cv_anchor_candidates_per_split=4 \
    --psd_cv_anchor_num_splits=3 \
    --psd_cv_anchor_training_rows=200 \
    --psd_cv_anchor_newton_iterations=5 \
    --psd_cv_anchor_metric=ed \
    --verbose \
    --verbose_newton \
    --psd_verbose_metrics


