#!/bin/bash
#SBATCH --job-name=pspstain_val_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_validate_BCI.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_validate_BCI.%j.err

# train_validate_BCI.sh
# Runs 3 epochs of PSPStain training on BCI as a cluster confirmation gate.
# This job must pass before submitting the full training jobs.
#
# Submit: sbatch train_validate_BCI.sh
#
# Pass criteria:
#   1. Job exits cleanly (no Python traceback in log)
#   2. Loss values in log are not NaN
#   3. Checkpoint files exist after the job:
#        find $VSC_DATA/projects/pspstain/outputs/checkpoints/BCI_validate_e3 -name "*.pth"
#   4. GPU log CSV has entries:
#        tail -5 $VSC_DATA/projects/pspstain/logs/gpu_train_validate_BCI.csv

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/pspstain_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/pspstain/code/pspstain"
CHECKPOINTS_DIR="$VSC_DATA/projects/pspstain/outputs/checkpoints"
RUN_NAME="BCI_validate_e3"

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
echo "=== Dataset check ==="
apptainer exec \
    -B "$VSC_SCRATCH/BCI-AB.sqsh:/data/BCI-AB:image-src=/" \
    "$CONTAINER" \
    bash -c 'echo "  trainA: $(ls /data/BCI-AB/trainA | wc -l) images"; echo "  trainB: $(ls /data/BCI-AB/trainB | wc -l) images"; echo "  valA:   $(ls /data/BCI-AB/valA | wc -l) images"; echo "  valB:   $(ls /data/BCI-AB/valB | wc -l) images"'

echo ""
echo "=== Repo check ==="
if [ ! -f "$REPO_DIR/train.py" ]; then
    echo "ERROR: train.py not found in $REPO_DIR"
    exit 1
fi
echo "  train.py found"

echo ""
echo "=== UNet weights check ==="
if [ ! -f "$REPO_DIR/pretrain/BCI_unet_seg.pth" ]; then
    echo "ERROR: pretrain/BCI_unet_seg.pth not found"
    exit 1
fi
echo "  BCI_unet_seg.pth found"

mkdir -p "$CHECKPOINTS_DIR/$RUN_NAME"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/pspstain/logs/gpu_train_validate_BCI.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting validation training (3 epochs, BCI) ==="
echo "  run name    : $RUN_NAME"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"

apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$VSC_SCRATCH/BCI-AB.sqsh:/data/BCI-AB:image-src=/" \
    "$CONTAINER" \
    python train.py \
        --dataroot /data/BCI-AB \
        --name "$RUN_NAME" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --model PSPStain \
        --CUT_mode FastCUT \
        --netG resnet_6blocks \
        --netD n_layers \
        --n_layers_D 5 \
        --ndf 32 \
        --normG instance \
        --normD instance \
        --weight_norm spectral \
        --dataset_mode aligned \
        --direction AtoB \
        --load_size 512 \
        --crop_size 512 \
        --batch_size 1 \
        --n_epochs 3 \
        --n_epochs_decay 0 \
        --save_epoch_freq 1 \
        --display_id 0 \
        --no_html \
        --unet_seg BCI_unet_seg \
        --gpu_ids 0

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Post-run checkpoint check ==="
find "$CHECKPOINTS_DIR/$RUN_NAME" -name "*.pth" | sort

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/pspstain/logs/gpu_train_validate_BCI.csv"

echo ""
echo "Validation training complete. Review the output above before submitting full runs."
