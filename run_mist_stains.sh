#!/bin/bash

conda activate PSPStain

STAINS=("HER2" "Ki67" "ER" "PR")

for STAIN in "${STAINS[@]}"; do
  echo "===================================="
  echo "Running PSPStain on MIST - ${STAIN}"
  echo "===================================="

  python test.py \
    --dataroot /home/thomas/Downloads/MIST/${STAIN}/TrainValAB \
    --name PSPStain_MIST \
    --model PSPStain \
    --checkpoints_dir checkpoints/PSPStain \
    --phase val \
    --dataset_mode aligned \
    --direction AtoB \
    --num_test 1000
done

