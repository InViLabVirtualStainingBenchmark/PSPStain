#!/bin/bash
#SBATCH --job-name=pspstain_infer_MIST_512_e100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/pspstain/logs/infer_MIST_512_e100.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/pspstain/logs/infer_MIST_512_e100.%j.err

# infer_MIST_512_e100.sh
# Runs inference on the full MIST-HER2 val split using the latest checkpoint
# from the MIST-HER2 100-epoch training run.
#
# NOTE: PSPStain uses --phase val, not --phase test. Input images are
# read from valA/ and outputs are written to val_latest/images/fake_B/.
#
# Submit ONLY after train_MIST_512_e100.sh has completed successfully.
# Submit: sbatch infer_MIST_512_e100.sh
#
# Output images land at:
#   $VSC_DATA/projects/pspstain/outputs/results/MIST-HER2_512_e100/val_latest/images/fake_B/

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/pspstain_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/pspstain/code/pspstain"
CHECKPOINTS_DIR="$VSC_DATA/projects/pspstain/outputs/checkpoints"
RESULTS_DIR="$VSC_DATA/projects/pspstain/outputs/results"
RUN_NAME="MIST-HER2_512_e100"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
MIST_MNT="$VSC_SCRATCH/sqsh_mnt/MIST-HER2"

# =========================
# MODULES
# =========================

module purge
module load calcua/2026.1

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Container ==="
echo "  $CONTAINER"
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

echo ""
echo "=== Environment ==="
apptainer exec --nv "$CONTAINER" python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "=== Checkpoint check ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint folder not found: $CKPT_DIR"
    echo "Has train_MIST_512_e100.sh completed successfully?"
    exit 1
fi
echo "  Checkpoints found:"
find "$CKPT_DIR" -name "*.pth" | sort

echo ""
echo "=== Val dataset check ==="
mkdir -p "$MIST_MNT"
apptainer exec \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  valA: \$(ls $MIST_MNT/valA | wc -l) images\""

echo ""
echo "=== UNet weights check ==="
if [ ! -f "$REPO_DIR/pretrain/MIST_unet_seg.pth" ]; then
    echo "ERROR: pretrain/MIST_unet_seg.pth not found"
    exit 1
fi
echo "  MIST_unet_seg.pth found"

mkdir -p "$RESULTS_DIR/$RUN_NAME"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/pspstain/logs/gpu_infer_MIST_512_e100.csv" & GPU_LOG_PID=$!

# =========================
# INFERENCE
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST-HER2 inference ==="
echo "  run name    : $RUN_NAME"
echo "  results dir : $RESULTS_DIR/$RUN_NAME"
echo "  dataroot    : $MIST_MNT (inside MIST-HER2.sqsh)"

apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    python test.py \
        --dataroot     "$MIST_MNT" \
        --name         "$RUN_NAME" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --results_dir  "$RESULTS_DIR" \
        --model        PSPStain \
        --CUT_mode     FastCUT \
        --netG         resnet_6blocks \
        --netD         n_layers \
        --n_layers_D   5 \
        --ndf          32 \
        --normG        instance \
        --normD        instance \
        --weight_norm  spectral \
        --dataset_mode aligned \
        --direction    AtoB \
        --load_size    512 \
        --crop_size    512 \
        --phase        val \
        --num_test     9999 \
        --no_flip \
        --unet_seg     MIST_unet_seg \
        --gpu_ids      0

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Output image count ==="
find "$RESULTS_DIR/$RUN_NAME" -name "*.png" | wc -l

echo ""
echo "=== Output folder structure ==="
ls "$RESULTS_DIR/$RUN_NAME/val_latest/images/" 2>/dev/null || echo "WARNING: val_latest/images/ not found"

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/pspstain/logs/gpu_infer_MIST_512_e100.csv"

echo ""
echo "MIST-HER2 inference complete. Next step: sbatch eval_MIST_512_e100.sh"