#!/bin/bash
#SBATCH --job-name=pspstain_BCI_1024crop_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=14:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_BCI_1024crop_p2.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_BCI_1024crop_p2.%j.err

# train_BCI_1024crop_e100_part2.sh
# Epochs 51-100 of PSPStain training on BCI at 512x512.
# Resumes from the latest checkpoint saved by part 1.
# Linear LR decay applied across these 50 epochs (n_epochs_decay=50).
#
# The LR schedule is equivalent to a single 100-epoch run:
#   Part 1: epochs  1-50  constant LR  (n_epochs=50, n_epochs_decay=0)
#   Part 2: epochs 51-100 linear decay (epoch_count=51, n_epochs=50, n_epochs_decay=50)
#
# DO NOT submit this manually before part 1 finishes -- use submit_BCI_e100.sh.
# If part 1 failed or was cancelled, do not submit this script.
#
# Monitor:
#   squeue -u $USER
#   tail -f $VSC_DATA/projects/pspstain/logs/train_BCI_1024crop_p2.<jobid>.out
#
# After this job completes, next step: sbatch infer_BCI_1024crop_e100.sh

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/pspstain_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/pspstain/code/pspstain"
CHECKPOINTS_DIR="$VSC_DATA/projects/pspstain/outputs/checkpoints"
RUN_NAME="BCI_1024crop_e100"
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
echo "=== Checkpoint check (part 1 must have completed) ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -f "$CKPT_DIR/latest_net_G.pth" ]; then
    echo "ERROR: latest_net_G.pth not found in $CKPT_DIR"
    echo "Has part 1 completed successfully?"
    exit 1
fi
echo "  latest checkpoint found:"
find "$CKPT_DIR" -name "latest_net_*.pth" | sort

echo ""
echo "=== UNet weights check ==="
if [ ! -f "$REPO_DIR/pretrain/BCI_unet_seg.pth" ]; then
    echo "ERROR: pretrain/BCI_unet_seg.pth not found"
    exit 1
fi
echo "  BCI_unet_seg.pth found"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/pspstain/logs/gpu_train_BCI_1024crop_p2.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting BCI training part 2 (epochs 51-100, LR decay) ==="
echo "  run name    : $RUN_NAME"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"
echo "  dataroot    : $BCI_AB_MNT (inside BCI-AB.sqsh)"

srun apptainer exec --nv \
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
        --load_size       1024 \
        --crop_size       512 \
        --preprocess      crop \
        --batch_size      1 \
        --n_epochs        50 \
        --n_epochs_decay  50 \
        --epoch_count     51 \
        --continue_train True \
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
tail -3 "$VSC_DATA/projects/pspstain/logs/gpu_train_BCI_1024crop_p2.csv"

echo ""
echo "BCI full training complete (epochs 1-100). Next step: sbatch infer_BCI_1024crop_e100.sh"
