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
# NOTE: The cluster's Apptainer auto-binds the host /data/ filesystem into every
# container, masking any /data/ subdirectories created during the container build.
# Squashfs archives are therefore mounted to paths under $VSC_SCRATCH instead.
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
BCI_AB_SQSH="$VSC_SCRATCH/BCI-AB.sqsh"
BCI_AB_MNT="$VSC_SCRATCH/sqsh_mnt/BCI-AB"

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
echo "=== SquashFS check ==="
if [ ! -f "$BCI_AB_SQSH" ]; then
    echo "ERROR: BCI-AB squashfs not found: $BCI_AB_SQSH"
    exit 1
fi
echo "  BCI-AB.sqsh found"

echo ""
echo "=== Dataset check ==="
mkdir -p "$BCI_AB_MNT"
apptainer exec \
    -B "$BCI_AB_SQSH:$BCI_AB_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $BCI_AB_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $BCI_AB_MNT/trainB | wc -l) images\"; echo \"  valA:   \$(ls $BCI_AB_MNT/valA | wc -l) images\"; echo \"  valB:   \$(ls $BCI_AB_MNT/valB | wc -l) images\""

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
echo "=== Starting validation training (3 epochs, BCI, load 1024, crop 512) ==="
echo "  run name    : $RUN_NAME"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"
echo "  dataroot    : $BCI_AB_MNT (inside BCI-AB.sqsh)"

apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$BCI_AB_SQSH:$BCI_AB_MNT:image-src=/" \
    "$CONTAINER" \
    python train.py \
        --dataroot     "$BCI_AB_MNT" \
        --name         "$RUN_NAME" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
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
        --load_size    1024 \
        --crop_size    512 \
        --preprocess   crop \
        --batch_size   1 \
        --n_epochs     3 \
        --n_epochs_decay 0 \
        --save_epoch_freq 1 \
        --display_id   0 \
        --no_html \
        --unet_seg     BCI_unet_seg \
        --gpu_ids      0

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