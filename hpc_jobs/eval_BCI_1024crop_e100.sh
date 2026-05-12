#!/bin/bash
#SBATCH --job-name=pspstain_eval_BCI_1024crop_e100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/pspstain/logs/eval_BCI_1024crop_e100.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/pspstain/logs/eval_BCI_1024crop_e100.%j.err

# eval_BCI_1024crop_e100.sh
# Runs evaluate.py on PSPStain BCI inference outputs using the shared
# evaluate_nvidia.sif container.
#
# Submit ONLY after infer_BCI_1024crop_e100.sh has completed and image count is correct.
# Submit: sbatch eval_BCI_1024crop_e100.sh
#
# Results appended to:
#   $VSC_DATA/benchmark_results.csv

set -euo pipefail

EVAL_CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
RESULTS_DIR="$VSC_DATA/projects/pspstain/outputs/results"
RUN_NAME="BCI_1024crop_e100"
PRED_DIR="$RESULTS_DIR/$RUN_NAME/val_latest/images/fake_B"
BCI_AB_SQSH="$VSC_SCRATCH/BCI-AB.sqsh"
BCI_AB_MNT="$VSC_SCRATCH/sqsh_mnt/BCI-AB"
GT_DIR="$BCI_AB_MNT/valB"

# =========================
# MODULES
# =========================

module purge
module load calcua/2026.1

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Evaluate container ==="
if [ ! -f "$EVAL_CONTAINER" ]; then
    echo "ERROR: evaluate_nvidia.sif not found: $EVAL_CONTAINER"
    exit 1
fi
echo "  found"

echo ""
echo "=== SquashFS check ==="
if [ ! -f "$BCI_AB_SQSH" ]; then
    echo "ERROR: BCI-AB squashfs not found: $BCI_AB_SQSH"
    exit 1
fi
echo "  BCI-AB.sqsh found"

echo ""
echo "=== Prediction folder check ==="
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Prediction folder not found: $PRED_DIR"
    echo "Has infer_BCI_1024crop_e100.sh completed successfully?"
    exit 1
fi
echo "  fake_B images: $(find "$PRED_DIR" -name "*.png" | wc -l)"

# =========================
# EVALUATION
# =========================

mkdir -p "$BCI_AB_MNT"

echo ""
echo "=== Starting BCI evaluation ==="
echo "  predictions : $PRED_DIR"
echo "  ground truth: $GT_DIR (inside BCI-AB.sqsh)"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$BCI_AB_SQSH:$BCI_AB_MNT:image-src=/" \
    "$EVAL_CONTAINER" \
    python "$VSC_DATA/evaluate/evaluate.py" \
        --pred         "$PRED_DIR" \
        --gt           "$GT_DIR" \
        --model_name   PSPStain \
        --dataset_name BCI \
        --split_name   val \
        --match_by     stem \
        --output       "$VSC_DATA/benchmark_results.csv" \
        --cellpose \
        --cellpose_model cpsam \
        --cellpose_n   100

# =========================
# POST-RUN REPORT
# =========================

echo ""
echo "=== benchmark_results.csv (last 3 rows) ==="
tail -3 "$VSC_DATA/benchmark_results.csv"

echo ""
echo "BCI evaluation complete."