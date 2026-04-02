#!/usr/bin/env python3
"""
run_mist_inference.py
---------------------
Runs PSPStain inference on all four MIST stains using the SINGLE shared
checkpoint at:  checkpoints/mist/latest_net_G.pth

Usage (from ~/PSPStain/):
    python run_mist_inference.py

Outputs land in:
    results/MIST_HER2/mist/val_latest/images/
    results/MIST_Ki67/mist/val_latest/images/
    results/MIST_ER/mist/val_latest/images/
    results/MIST_PR/mist/val_latest/images/
"""

import subprocess
import sys
import os
from pathlib import Path

# ── USER SETTINGS ─────────────────────────────────────────────────────────────

PSPSTAIN_ROOT   = Path.home() / "PSPStain"           # where test.py lives
MIST_ROOT       = Path.home() / "Downloads" / "MIST" # parent of HER2/Ki67/ER/PR

CHECKPOINTS_DIR = PSPSTAIN_ROOT / "checkpoints"      # contains: mist/latest_net_G.pth
CHECKPOINT_NAME = "mist"                              # folder name inside checkpoints/

STAINS = ["HER2", "Ki67", "ER", "PR"]

# ── SANITY CHECK ──────────────────────────────────────────────────────────────

def check_prereqs():
    test_py = PSPSTAIN_ROOT / "test.py"
    ckpt    = CHECKPOINTS_DIR / CHECKPOINT_NAME / "latest_net_G.pth"

    if not test_py.exists():
        sys.exit(f"[ERROR] test.py not found at {test_py}\n"
                 f"        Make sure PSPSTAIN_ROOT is correct.")

    if not ckpt.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt}\n"
                 f"        Expected: checkpoints/mist/latest_net_G.pth")

    for stain in STAINS:
        val_a = MIST_ROOT / stain / "TrainValAB" / "valA"
        if not val_a.exists():
            sys.exit(f"[ERROR] valA folder missing for stain '{stain}': {val_a}")

    print("[OK] All prerequisites found.\n")

# ── INFEOOP ────────────────────────────────────────────────────────────

def run_inference():
    failures = []

    for stain in STAINS:
        dataroot    = MIST_ROOT / stain / "TrainValAB"
        results_dir = PSPSTAIN_ROOT / "results" / f"MIST_{stain}"

        results_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(PSPSTAIN_ROOT / "test.py"),
            "--dataroot",        str(dataroot),
            "--name",            CHECKPOINT_NAME,
            "--checkpoints_dir", str(CHECKPOINTS_DIR),
            "--results_dir",     str(results_dir),
            "--model",           "PSPStain",
            "--netG",            "resnet_6blocks",
            "--netD",            "n_layers",
            "--n_layers_D",      "5",
            "--ndf",             "32",
            "--weight_norm",     "spectral",
            "--dataset_mode",    "aligned",
            "--direction",       "AtoB",
            "--phase",           "val",
            "--epoch",           "latest",
            "--crop_size",       "1024",
            "--load_size",       "1024",
            "--num_test",        "10000",
            "--eval",
            "--no_flip",
            "--gpu_ids",         "0",
        ]


        print("=" * 65)
        print(f"  STAIN : {stain}")
        print(f"  DATA  : {dataroot}")
        print(f"  OUTPUT: {results_dir}/{CHECKPOINT_NAME}/val_latest/images/")
        print("=" * 65)
        print("CMD:", " ".join(cmd), "\n")

        result = subprocess.run(cmd, cwd=str(PSPSTAIN_ROOT), check=False)

        if result.returncode == 0:
            img_dir = results_dir / CHECKPOINT_NAME / "val_latest" / "images"
            imgs    = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
            print(f"\n[✓] {stain} done — {len(imgs)} image(s) saved to {img_dir}\n")
        else:
            print(f"\n[✗] {stain} FAILED with exit code {result.returncode}\n")
            failures.append(stain)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if not failures:
        print("  ALL 4 stains completed successfully ✓")
        print("\n  Results are in:")
        for stain in STAINS:
            p = PSPSTAIN_ROOT / "results" / f"MIST_{stain}" / CHECKPOINT_NAME / "val_latest" / "images"
            print(f"    {p}")
    else:
        ok = [s for s in STAINS if s not in failures]
        print(f"  Completed : {ok}")
        print(f"  Failed    : {failures}")
        print("\n  See error output above for details.")
    print("=" * 65)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_prereqs()
    run_inference()

