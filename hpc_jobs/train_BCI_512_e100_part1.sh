#!/bin/bash
#SBATCH --job-name=pspstain_BCI_512_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=14:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_BCI_512_p1.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_BCI_512_p1.%j.err

# train_BCI_512_e100_part1.sh
# Epochs 1-50 of PSPStain training on BCI at 512x512.
# Constant LR throughout (n_epochs_decay=0).
# BCI has 3896 training images -- ~14.5 min/epoch, too slow for 100 epochs in one 24h job.
# This is part 1 of 2. Part 2 resumes from the latest checkpoint and applies LR decay.
#
# DO NOT submit this manually -- use submit_BCI_e100.sh which chains both parts automatically.
#
# Monitor:
#   squeue -u $USER
#   tail -f $VSC_DATA/projects/pspstain/logs/train_BCI_512_p1.<jobid>.out
#
# Checkpoints saved at epoch 25 and epoch 50 (plus latest after every epoch):
#   $VSC_DATA/projects/pspstain/outputs/checkpoints/BCI_512_e100/

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/pspstain_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/pspstain/code/pspstain"
CHECKPOINTS_DIR="$VSC_DATA/projects/pspstain/outputs/checkpoints"
RUN_NAME="BCI_512_e100"
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
echo "=== Dataset check ==="
mkdir -p "$BCI_AB_MNT"
apptainer exec \
    -B "$BCI_AB_SQSH:$BCI_AB_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $BCI_AB_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $BCI_AB_MNT/trainB | wc -l) images\""

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
    > "$VSC_DATA/projects/pspstain/logs/gpu_train_BCI_512_p1.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting BCI training part 1 (epochs 1-50, constant LR) ==="
echo "  run name    : $RUN_NAME"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"
echo "  dataroot    : $BCI_AB_MNT (inside BCI-AB.sqsh)"

apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$BCI_AB_SQSH:$BCI_AB_MNT:image-src=/" \
    "$CONTAINER" \
    python train.py \
        --dataroot        "$BCI_AB_MNT" \
        --name            "$RUN_NAME" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --model           PSPStain \
        --CUT_mode        FastCUT \
        --netG            resnet_6blocks \
        --netD            n_layers \
        --n_layers_D      5 \
        --ndf             32 \
        --normG           instance \
        --normD           instance \
        --weight_norm     spectral \
        --dataset_mode    aligned \
        --direction       AtoB \
        --load_size       512 \
        --crop_size       512 \
        --batch_size      1 \
        --n_epochs        50 \
        --n_epochs_decay  0 \
        --save_epoch_freq 25 \
        --display_id      0 \
        --no_html \
        --unet_seg        BCI_unet_seg \
        --gpu_ids         0

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Post-run checkpoint check ==="
find "$CHECKPOINTS_DIR/$RUN_NAME" -name "*.pth" | sort

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/pspstain/logs/gpu_train_BCI_512_p1.csv"

echo ""
echo "BCI part 1 complete (epochs 1-50). Part 2 should start automatically if submitted via wrapper."
