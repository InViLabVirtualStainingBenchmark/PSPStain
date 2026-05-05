#!/bin/bash
#SBATCH --job-name=pspstain_train_BCI_512_e100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=XX:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_BCI_512_e100.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_BCI_512_e100.%j.err

# train_BCI_512_e100.sh
# Full 100-epoch PSPStain training on the BCI dataset.
# 50 epochs constant LR + 50 epochs linear LR decay = 100 total.
#
# TODO: Replace XX:00:00 above with the correct wall time before submitting.
#       Use the time-per-epoch from the validate job log to estimate:
#       (seconds_per_epoch * 100 * 1.20) / 3600 rounded up to the next hour.
#
# Submit ONLY after train_validate_BCI.sh has passed all criteria.
# Submit: sbatch train_BCI_512_e100.sh
#
# Checkpoints saved at epochs 25, 50, 75, and 100 (plus latest after every epoch):
#   $VSC_DATA/projects/pspstain/outputs/checkpoints/BCI_full_e100/

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/pspstain_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/pspstain/code/pspstain"
CHECKPOINTS_DIR="$VSC_DATA/projects/pspstain/outputs/checkpoints"
RUN_NAME="BCI_512_e100"

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
    bash -c 'echo "  trainA: $(ls /data/BCI-AB/trainA | wc -l) images"; echo "  trainB: $(ls /data/BCI-AB/trainB | wc -l) images"'

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
    > "$VSC_DATA/projects/pspstain/logs/gpu_train_BCI_512_e100.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting full BCI training (100 epochs) ==="
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
        --n_epochs 50 \
        --n_epochs_decay 50 \
        --save_epoch_freq 25 \
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
tail -3 "$VSC_DATA/projects/pspstain/logs/gpu_train_BCI_512_e100.csv"

echo ""
echo "BCI full training complete. Next step: sbatch infer_BCI_full.sh"
