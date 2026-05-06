#!/bin/bash
# submit_MIST_e100.sh
# Submission wrapper for the two-part MIST-HER2 100-epoch training.
# Submits part 1, then submits part 2 with a SLURM dependency so that
# part 2 only starts if part 1 exits cleanly (no crash, no timeout).
#
# Usage (run from the repo root on the login node):
#   bash hpc_jobs/submit_MIST_e100.sh
#
# What happens:
#   1. Part 1 is submitted immediately and queued normally.
#   2. Part 2 is submitted with --dependency=afterok:<part1_jobid>.
#      It will sit in state "Pending (Dependency)" until part 1 finishes.
#      If part 1 fails or is cancelled, part 2 is automatically cancelled too.
#
# To check status:
#   squeue -u $USER
#
# To cancel both jobs if needed:
#   scancel <part1_jobid> <part2_jobid>
#
# After both jobs finish, submit inference:
#   sbatch hpc_jobs/512_e100/infer_MIST_512_e100.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART1="$SCRIPT_DIR/train_MIST_512_e100_part1.sh"
PART2="$SCRIPT_DIR/train_MIST_512_e100_part2.sh"

if [ ! -f "$PART1" ]; then
    echo "ERROR: $PART1 not found."
    exit 1
fi
if [ ! -f "$PART2" ]; then
    echo "ERROR: $PART2 not found."
    exit 1
fi

echo "=== Submitting MIST-HER2 100-epoch training (2 parts) ==="

JOB1=$(sbatch --parsable "$PART1")
echo "  Part 1 submitted: job $JOB1  (epochs 1-50, constant LR)"

JOB2=$(sbatch --parsable --dependency=afterok:"$JOB1" "$PART2")
echo "  Part 2 submitted: job $JOB2  (epochs 51-100, LR decay)"
echo "  Part 2 will only start if part 1 exits successfully."

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Part 1 log:   tail -f \$VSC_DATA/projects/pspstain/logs/train_MIST_512_p1.$JOB1.out"
echo "Part 2 log:   tail -f \$VSC_DATA/projects/pspstain/logs/train_MIST_512_p2.$JOB2.out"
echo ""
echo "After both complete, submit inference:"
echo "  sbatch hpc_jobs/512_e100/infer_MIST_512_e100.sh"
