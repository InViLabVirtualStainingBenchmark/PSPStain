import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

# Metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance

# ---------------------------------------------------------
# Optional metrics (AUC, Dice) — safely imported
# ---------------------------------------------------------

# AUC (optional)
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

# Dice (optional)
try:
    from torchmetrics import DiceCoefficient
    dice_metric = DiceCoefficient()
except ImportError:
    dice_metric = None

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def to_tensor(img):
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)

# ---------------------------------------------------------
# Metric initializers
# ---------------------------------------------------------

lpips_fn = lpips.LPIPS(net='alex')
fid = FrechetInceptionDistance(feature=64)

# ---------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------

def evaluate_metrics(gt_dir, pred_dir, save_csv="metrics_results.csv"):
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))
    pred_paths = sorted(glob(os.path.join(pred_dir, "*.png")))

    assert len(gt_paths) == len(pred_paths), "Mismatch in number of images"

    results = []

    for gt_path, pred_path in tqdm(zip(gt_paths, pred_paths), total=len(gt_paths)):
        gt = load_image(gt_path)
        pred = load_image(pred_path)

        # SSIM + PSNR
        ssim_val = ssim(gt, pred, channel_axis=2, data_range=255)
        psnr_val = psnr(gt, pred, data_range=255)

        # LPIPS (float tensors)
        gt_t = to_tensor(gt)
        pred_t = to_tensor(pred)
        lpips_val = lpips_fn(gt_t, pred_t).item()

        # FID (uint8 tensors)
        gt_uint8 = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
        pred_uint8 = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)

        fid.update(gt_uint8, real=True)
        fid.update(pred_uint8, real=False)

        results.append({
            "image": os.path.basename(gt_path),
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "LPIPS": lpips_val,
        })

    # Compute FID at the end
    try:
        fid_score = fid.compute().item()
        print(f"\nFinal FID: {fid_score}")
    except RuntimeError as e:
        print("\n[WARNING] FID could not be computed:")
        print("  →", str(e))
        print("  → You need at least 2 real and 2 fake images.")
        fid_score = None

    df = pd.DataFrame(results)
    df["FID"] = fid_score
    df.to_csv(save_csv, index=False)

    print(f"Saved results to {save_csv}")
    generate_html_dashboard(df,gt_dir,pred_dir)

    return df



def generate_html_dashboard(df, gt_dir, pred_dir, output_path="metrics_dashboard.html"):
    html = """
    <html>
    <head>
        <title>PSPStain Evaluation Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; cursor: pointer; }
            img { width: 128px; height: 128px; object-fit: cover; transition: 0.2s; }
            img:hover { transform: scale(2.5); z-index: 10; }
            .row { display: flex; gap: 20px; margin-bottom: 20px; }
            .summary-box { padding: 10px; border: 1px solid #ccc; border-radius: 5px; width: 200px; }
        </style>
        <script>
            function sortTable(n) {
                var table = document.getElementById("metricsTable");
                var rows = table.rows;
                var switching = true;
                var dir = "asc";
                var switchcount = 0;

                while (switching) {
                    switching = false;
                    var shouldSwitch;

                    for (var i = 1; i < rows.length - 1; i++) {
                        shouldSwitch = false;
                        var x = rows[i].getElementsByTagName("TD")[n];
                        var y = rows[i + 1].getElementsByTagName("TD")[n];

                        var xVal = parseFloat(x.innerHTML) || x.innerHTML.toLowerCase();
                        var yVal = parseFloat(y.innerHTML) || y.innerHTML.toLowerCase();

                        if (dir == "asc" && xVal > yVal) { shouldSwitch = true; break; }
                        if (dir == "desc" && xVal < yVal) { shouldSwitch = true; break; }
                    }

                    if (shouldSwitch) {
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    } else {
                        if (switchcount == 0 && dir == "asc") {
                            dir = "desc";
                            switching = true;
                        }
                    }
                }
            }
        </script>
    </head>
    <body>
        <h1>PSPStain Evaluation Dashboard</h1>
        <h3>Interactive metrics + image viewer</h3>
    """

    # Summary statistics
    html += "<div class='row'>"
    for metric in ["SSIM", "PSNR", "LPIPS"]:
        html += f"""
        <div class='summary-box'>
            <h4>{metric}</h4>
            <p>Mean: {df[metric].mean():.4f}</p>
            <p>Std: {df[metric].std():.4f}</p>
        </div>
        """
    html += "</div>"

    # Table header
    html += """
    <table id="metricsTable">
        <tr>
            <th onclick="sortTable(0)">Image</th>
            <th>Real</th>
            <th>Fake</th>
            <th onclick="sortTable(3)">SSIM</th>
            <th onclick="sortTable(4)">PSNR</th>
            <th onclick="sortTable(5)">LPIPS</th>
        </tr>
    """

    # Table rows
    for _, row in df.iterrows():
        img = row["image"]
        real_path = os.path.join(gt_dir, img)
        fake_path = os.path.join(pred_dir, img)

        html += f"""
        <tr>
            <td>{img}</td>
            <td><img src="{real_path}" /></td>
            <td><img src="{fake_path}" /></td>
            <td>{row['SSIM']:.4f}</td>
            <td>{row['PSNR']:.4f}</td>
            <td>{row['LPIPS']:.4f}</td>
        </tr>
        """

    html += "</table></body></html>"

    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nInteractive dashboard saved to: {output_path}")


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--save_csv", type=str, default="metrics_results.csv")

    args = parser.parse_args()

    evaluate_metrics(args.gt_dir, args.pred_dir, args.save_csv)
