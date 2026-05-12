#!/bin/bash
#SBATCH --job-name=pspstain_MIST_1024crop_p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:30:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_MIST_1024crop_p3.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/pspstain/logs/train_MIST_1024crop_p3.%j.err

# train_MIST_1024crop_e100_part3.sh
# Epochs 99-100 of PSPStain training on MIST-HER2 at 512x512.
# Resumes from the latest checkpoint saved by part 2 (epoch 98, total_iters 220000).
#
# Part 2 was cancelled by the cluster time limit at the end of epoch 98.
# The loop range is determined by:
#   range(epoch_count, n_epochs + n_epochs_decay + 1)
#   range(99, 50 + 50 + 1) = range(99, 101) => epochs 99 and 100
#
# LR schedule context (full 100-epoch run):
#   Part 1: epochs  1-50   constant LR  (n_epochs=50, n_epochs_decay=0)
#   Part 2: epochs 51-100  linear decay (epoch_count=51, n_epochs=50, n_epochs_decay=50)
#   Part 3: epochs 99-100  same decay params, epoch_count=99 to restrict the loop
#
# Note: because the LR scheduler is re-initialised from the saved optimiser state,
# the decay will be approximately correct for these final two epochs (~0.002 base LR).
# The difference is negligible at this stage of training.
#
# DO NOT submit this before part 2 has completed (or been cancelled with checkpoints intact).
#
# Monitor:
#   squeue -u $USER
#   tail -f $VSC_DATA/projects/pspstain/logs/train_MIST_1024crop_p3.<jobid>.out
#
# After this job completes, next step: sbatch infer_MIST_1024crop_e100.sh

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/pspstain_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/pspstain/code/pspstain"
CHECKPOINTS_DIR="$VSC_DATA/projects/pspstain/outputs/checkpoints"
RUN_NAME="MIST-HER2_1024crop_e100"
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
echo "=== Dataset check ==="
mkdir -p "$MIST_MNT"
apptainer exec \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $MIST_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $MIST_MNT/trainB | wc -l) images\""

echo ""
echo "=== Checkpoint check (part 2 epoch 98 must exist) ==="
CKPT_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
if [ ! -f "$CKPT_DIR/latest_net_G.pth" ]; then
    echo "ERROR: latest_net_G.pth not found in $CKPT_DIR"
    echo "Has part 2 completed or saved at least one checkpoint?"
    exit 1
fi
echo "  latest checkpoint found:"
find "$CKPT_DIR" -name "latest_net_*.pth" | sort
echo ""
echo "  all checkpoints in run directory:"
find "$CKPT_DIR" -name "*.pth" | sort

echo ""
echo "=== UNet weights check ==="
if [ ! -f "$REPO_DIR/pretrain/MIST_unet_seg.pth" ]; then
    echo "ERROR: pretrain/MIST_unet_seg.pth not found"
    exit 1
fi
echo "  MIST_unet_seg.pth found"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/pspstain/logs/gpu_train_MIST_1024crop_p3.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST-HER2 training part 3 (epochs 99-100, final LR decay) ==="
echo "  run name    : $RUN_NAME"
echo "  checkpoints : $CHECKPOINTS_DIR/$RUN_NAME"
echo "  dataroot    : $MIST_MNT (inside MIST-HER2.sqsh)"
echo "  resuming from latest checkpoint (epoch 98)"

apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_SQSH:$MIST_MNT:image-src=/" \
    "$CONTAINER" \
    python train.py \
        --dataroot        "$MIST_MNT" \
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
        --epoch_count     99 \
        --continue_train True \
        --save_epoch_freq 1 \
        --display_id      0 \
        --no_html \
        --unet_seg        MIST_unet_seg \
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
tail -3 "$VSC_DATA/projects/pspstain/logs/gpu_train_MIST_1024crop_p3.csv"

echo ""
echo "MIST-HER2 full training complete (epochs 1-100). Next step: sbatch infer_MIST_1024crop_e100.sh"