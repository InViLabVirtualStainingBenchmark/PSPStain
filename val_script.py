import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr_ssim(pred_dir, real_dir):
    """
    Compute PSNR and SSIM between predicted IHC (fake_B) and real IHC (real_B).
    """

    psnr_scores = []
    ssim_scores = []

    # Get all predicted files
    pred_files = sorted(os.listdir(pred_dir))

    for fname in pred_files:
        pred_path = os.path.join(pred_dir, fname)
        real_path = os.path.join(real_dir, fname)

        if not os.path.exists(real_path):
            print(f"[WARNING] Missing real image for {fname}")
            continue

        pred = cv2.imread(pred_path)
        real = cv2.imread(real_path)

        if pred is None or real is None:
            print(f"[ERROR] Could not read {fname}")
            continue

        # Convert BGR → RGB
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

        # Resize prediction to match real image
        pred = cv2.resize(pred, (real.shape[1], real.shape[0]))

        # Compute metrics
        psnr = peak_signal_noise_ratio(real, pred, data_range=255)
        ssim = structural_similarity(real, pred, channel_axis=2, data_range=255)

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    print("====================================")
    print(f"Images evaluated: {len(psnr_scores)}")

    if psnr_scores:
        print(f"Average PSNR: {np.mean(psnr_scores):.4f}")
        print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    else:
        print("No valid image pairs found.")

    print("====================================")


if __name__ == "__main__":
    pred_dir = "results/mist/val_latest/images/fake_B"
    real_dir = "results/mist/val_latest/images/real_B"

    compute_psnr_ssim(pred_dir, real_dir)
