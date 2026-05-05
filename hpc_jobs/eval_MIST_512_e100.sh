#!/bin/bash
#SBATCH --job-name=pspstain_eval_MIST_512_e100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/pspstain/logs/eval_MIST_512_e100.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/pspstain/logs/eval_MIST_512_e100.%j.err

# eval_MIST_512_e100.sh
# Runs evaluate.py on PSPStain MIST-HER2 inference outputs using the shared
# evaluate_nvidia.sif container.
#
# Submit ONLY after infer_MIST_512_e100.sh has completed and image count is correct.
# Submit: sbatch eval_MIST_512_e100.sh
#
# Results appended to:
#   $VSC_DATA/benchmark_results.csv

set -euo pipefail

EVAL_CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
RESULTS_DIR="$VSC_DATA/projects/pspstain/outputs/results"
RUN_NAME="MIST-HER2_512_e100"
PRED_DIR="$RESULTS_DIR/$RUN_NAME/val_latest/images/fake_B"

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
echo "=== Prediction folder check ==="
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Prediction folder not found: $PRED_DIR"
    echo "Has infer_MIST_full.sh completed successfully?"
    exit 1
fi
echo "  fake_B images: $(find "$PRED_DIR" -name "*.png" | wc -l)"

# =========================
# EVALUATION
# =========================

echo ""
echo "=== Starting MIST-HER2 evaluation ==="
echo "  predictions : $PRED_DIR"
echo "  ground truth: /data/MIST-HER2/valB"

apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$VSC_SCRATCH/MIST-HER2.sqsh:/data/MIST-HER2:image-src=/" \
    "$EVAL_CONTAINER" \
    python "$VSC_DATA/evaluate/evaluate.py" \
        --pred "$PRED_DIR" \
        --gt /data/MIST-HER2/valB \
        --model_name PSPStain \
        --dataset_name MIST-HER2 \
        --split_name full_e100 \
        --match_by stem \
        --output "$VSC_DATA/benchmark_results.csv"

# =========================
# POST-RUN REPORT
# =========================

echo ""
echo "=== benchmark_results.csv (last 3 rows) ==="
tail -3 "$VSC_DATA/benchmark_results.csv"

echo ""
echo "MIST-HER2 evaluation complete."
