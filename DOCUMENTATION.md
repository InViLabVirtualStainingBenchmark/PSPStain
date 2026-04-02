
##  PSPStain: Training, Inference & Evaluation Pipeline

---
> This document summarizes the complete workflow, environment setup, dataset mapping, training, inference, and evaluation steps required to run the **PSPStain** model on both the **BCI** and **MIST** datasets.



---

## 1. Hardware Used

- Ubuntu 22.04
- NVIDIA RTX 4080 (16GB VRAM)
- Python 3.9
- Conda environment: `PSPStain`

---

## 2. Requirements

### 2.1 Required Environment (from `environment.yml`)

The PSPStain repository was originally designed for the following stack:

| Component    | Required Version                              |
|--------------|-----------------------------------------------|
| Python       | 3.9                                           |
| PyTorch      | 1.12.1                                        |
| TorchVision  | 0.13.1                                        |
| CUDA Toolkit | 11.6                                          |
| Other deps   | scipy, dominate, Pillow, numpy, visdom, GPUtil |

### Environment Reconstruction
```bash
conda create -n PSPStain python=3.9 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda activate PSPStain
pip install dominate visdom GPUtil opencv-python
```

> 💡 **Important:** This environment is mandatory — PSPStain fails to train correctly on PyTorch 2.x or CUDA 12.x.

---

## 3. Infrastructure Fixes & Code Modernization

Several issues prevented PSPStain from running on modern systems. The following fixes were applied.

### 3.1 Evaluation Script Rewrite (`eval_metrics.py`)

The original repo had no evaluation pipeline. A complete evaluation suite was implemented from scratch, including:

- SSIM
- PSNR
- LPIPS (AlexNet)
- FID (torch-fidelity backend)

An interactive HTML dashboard is generated and saved as `metrics_dashboard.html`, which includes:
- Real vs Fake image thumbnails
- Hover-zoom
- Sortable metrics
- Summary statistics

### 3.2 Folder Structure Auto-Compatibility

PSPStain outputs results in:
```
results/<dataset>/<experiment>/val_latest/images/
    ├── real_A
    ├── real_B
    └── fake_B
```

Evaluation is standardized to use:
- Ground truth → `real_B`
- Predictions → `fake_B`

### 3.3 Checkpoint & `val_opt.txt` Mismatch Fix

The `val_opt.txt` shipped with the checkpoint was saved from a different training run and contained incorrect defaults:

| Option        | Wrong value (`val_opt.txt`) | Correct value (launcher) |
|---------------|-----------------------------|--------------------------|
| `model`       | `cut`                       | `PSPStain`               |
| `netG`        | `resnet_9blocks`            | `resnet_6blocks`         |
| `weight_norm` | `none`                      | `spectral`               |
| `ndf`         | `64`                        | `32`                     |
| `n_layers_D`  | `3`                         | `5`                      |
| `netD`        | `basic`                     | `n_layers`               |
| `crop_size`   | `512`                       | `1024`                   |
| `load_size`   | `512`                       | `1024`                   |

**Fix:** Always pass these arguments explicitly on the command line to override `val_opt.txt`.

---

## 4. Dataset

- Breast Cancer Immunohistochemical (BCI) challenge dataset
- Multi-IHC Stain Translation (MIST) dataset

More information and downloading links can be found in [BCI](https://bupt-ai-cz.github.io/BCI) and [MIST](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE).

### 4.1 Expected Folder Structure
```
dataset/
│
├── trainA/
│   └── HE/
├── trainB/
│   └── IHC/
├── valA/
└── valB/
```

### 4.2 BCI Dataset Mapping (H&E → IHC)

| Original Path | PSPStain Path        |
|---------------|----------------------|
| HE/train      | datasets/BCI/trainA  |
| IHC/train     | datasets/BCI/trainB  |
| HE/test       | datasets/BCI/testA   |
| IHC/test      | datasets/BCI/testB   |

> All filenames must match exactly.

### 4.3 MIST Dataset Mapping (H&E → Protein Modalities)

Each protein modality is its own dataset: `MIST_ER`, `MIST_PR`, `MIST_HER2`, `MIST_Ki67`

| Original Path | PSPStain Path |
|---------------|---------------|
| trainA        | trainA        |
| trainB        | trainB        |
| valA          | testA         |
| valB          | testB         |

---

## 5. Checkpoint

> 💡 **Important:** The checkpoint must be placed at `checkpoints/mist/latest_net_G.pth`.
> The file downloaded from the links below is named `MIST_net_G.pth` — rename it on placement:
```bash
mv ~/Downloads/MIST_net_G.pth ~/PSPStain/checkpoints/mist/latest_net_G.pth
```

- Baidu Drive [link](https://pan.baidu.com/s/1cPZ2Kk6JtURmtQhtNxzyEQ?pwd=u6qo) — key: `u6qo`
- Google Drive [link](https://drive.google.com/file/d/1BCqE_I5ZhLOrESryl2zAd39jU9fc6RvH/view?usp=drive_link)

---

## 6. Training from Scratch

We use `experiments/PSPStain_launcher.py` to generate the command line arguments for training and testing. More details on the parameters used in training our models can be found in that launcher file.

💡 **Important:** When training on a different dataset, change the pretrain UNet model to `pretrain/BCI_unet_seg.pth` or `pretrain/MIST_unet_seg.pth`.

### 6.1 BCI Training
```bash
python train.py \
    --dataroot datasets/BCI \
    --name PSPStain_BCI \
    --model PSPStain \
    --netG resnet_6blocks \
    --weight_norm spectral \
    --gpu_ids 0 \
    --batch_size 4 \
    --load_size 1024 \
    --crop_size 512 \
    --unet_seg BCI_unet_seg
```

Or using the launcher:
```bash
python -m experiments --name PSPStain --cmd train --id 0 --unet_seg 'BCI_unet_seg'
```

### 6.2 MIST Training (all stains)
```bash
python -m experiments --name PSPStain --cmd train --id 0 --unet_seg 'MIST_unet_seg'
```

---

## 7. Testing & Inference

### 7.1 Single Stain (smoke test — 5 images)
```bash
python test.py \
  --dataroot ~/Downloads/MIST/ER/TrainValAB \
  --name mist \
  --checkpoints_dir checkpoints \
  --results_dir ./results/test_ER \
  --model PSPStain \
  --netG resnet_6blocks \
  --netD n_layers \
  --n_layers_D 5 \
  --ndf 32 \
  --weight_norm spectral \
  --dataset_mode aligned \
  --direction AtoB \
  --phase val \
  --epoch latest \
  --crop_size 1024 \
  --load_size 1024 \
  --num_test 5 \
  --eval \
  --no_flip \
  --gpu_ids 0
```

### 7.2 All MIST Stains (HER2, Ki67, ER, PR)
```bash
python run_inference.py
```

This loops over all four stains automatically using the single shared checkpoint.

Results are saved to:
```
results/MIST_HER2/mist/val_latest/images/
results/MIST_Ki67/mist/val_latest/images/
results/MIST_ER/mist/val_latest/images/
results/MIST_PR/mist/val_latest/images/
```

### 7.3 Viewing Results
```bash
xdg-open ~/PSPStain/results/MIST_HER2/mist/val_latest/index.html
xdg-open ~/PSPStain/results/MIST_Ki67/mist/val_latest/index.html
xdg-open ~/PSPStain/results/MIST_ER/mist/val_latest/index.html
xdg-open ~/PSPStain/results/MIST_PR/mist/val_latest/index.html
```

---

## 8. Evaluation

### 8.1 Run Evaluation
```bash
python eval_metrics.py \
    --gt_dir results/MIST_ER/mist/val_latest/images/real_B \
    --pred_dir results/MIST_ER/mist/val_latest/images/fake_B
```

### 8.2 Outputs

- `metrics_results.csv` — per-image scores
- `metrics_dashboard.html` — interactive visual dashboard
- Terminal summary
- FID score

---

## Acknowledgement

This repo is built upon [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation) and [Adaptive Supervised PatchNCE Loss (ASP)](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE)
do it here
Hardware Used * Ubuntu 22.04 * RTX 4080 (16GB VRAM) * Python 3.9 * Conda environment: PSPStain 1.1 Required Environment (from environment.yml) The repository was originally designed for: Component Required Version Python 3.9 PyTorch 1.12.1 TorchVision 0.13.1 CUDA Toolkit 11.6 Other deps scipy, dominate, Pillow, numpy, visdom, GPUtil Environment Reconstruction conda create -n PSPStain python=3.9 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge conda activate PSPStain pip install dominate visdom GPUtil opencv-python This environment is mandatory — PSPStain fails to train correctly on PyTorch 2.x or CUDA 12.x. 2. Infrastructure Fixes & Code Modernization Several issues prevented PSPStain from running on modern systems. We applied the following fixes. 2.1 Evaluation Script Rewrite (eval_metrics.py) The original repo had no evaluation pipeline. We built a complete evaluation suite from scratch: ✔ Metrics implemented * SSIM * PSNR * LPIPS (AlexNet) * FID (torch‑fidelity backend) This dashboard is saved as: metrics_dashboard.html 2.2 Folder Structure Auto‑Compatibility PSPStain outputs results in: Code results/<dataset>/<experiment>/val_latest/images/     ├── real_A     ├── real_B     └── fake_B We standardized evaluation to use: * Ground truth → real_B * Predictions → fake_B 3. Dataset Mapping & Preparation PSPStain expects CycleGAN‑style aligned datasets: trainA / trainB testA / testB We mapped both datasets accordingly. 3.1 BCI Dataset (H&E → IHC) Original Path PSPStain Path HE/train datasets/BCI/trainA IHC/train datasets/BCI/trainB HE/test datasets/BCI/testA IHC/test datasets/BCI/testB All filenames must match exactly. 3.2 MIST Dataset (Unstained → Protein Modalities) Each protein modality becomes its own dataset: MIST_ER MIST_PR MIST_HER2 MIST_Ki67 4. Training Workflow 4.1 BCI Training bash python train.py \     --dataroot datasets/BCI \     --name PSPStain_BCI \     --gpu_ids 0 \     --batch_size 1 \     --load_size 256 \     --crop_size 256 4.2 MIST Training (Example: HER2) bash python train.py \     --dataroot datasets/MIST_HER2 \     --name PSPStain_MIST_HER2 \     --gpu_ids 0 \     --batch_size 1 \     --load_size 256 \     --crop_size 256 Repeat for ER, PR, Ki67. NB; To run the entire model on all the MIST data Subset(MIST_ER, MIST_PR, MIST_Ki67, MIST_HER2) we run the test run_inference.py file TO run the file ;  ./run_inference.py 6. Evaluation Workflow Run evaluation python eval_metrics.py     --gt_dir results/MIST_ER/mist/val_latest/images/real_B     --pred_dir results/MIST_ER/mist/val_latest/images/fake_B Outputs * metrics_results.csv * metrics_dashboard.html * Terminal summary * FID score
Show more
16:26
markdown
# PSPStain: Training, Inference & Evaluation Pipeline

---

## 1. Hardware Used

- Ubuntu 22.04
- NVIDIA RTX 4080 (16GB VRAM)
- Python 3.9
- Conda environment: `PSPStain`

### 1.1 Required Environment (from `environment.yml`)

The repository was originally designed for:

| Component    | Required Version                               |
|--------------|------------------------------------------------|
| Python       | 3.9                                            |
| PyTorch      | 1.12.1                                         |
| TorchVision  | 0.13.1                                         |
| CUDA Toolkit | 11.6                                           |
| Other deps   | scipy, dominate, Pillow, numpy, visdom, GPUtil |

### Environment Reconstruction
```bash
conda create -n PSPStain python=3.9 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda activate PSPStain
pip install dominate visdom GPUtil opencv-python
```

> ⚠️ This environment is mandatory — PSPStain fails to train correctly on PyTorch 2.x or CUDA 12.x.

---

## 2. Infrastructure Fixes & Code Modernization

Several issues prevented PSPStain from running on modern systems. We applied the following fixes.

### 2.1 Evaluation Script Rewrite (`eval_metrics.py`)

The original repo had no evaluation pipeline. We built a complete evaluation suite from scratch.

✔ **Metrics implemented:**
- SSIM
- PSNR
- LPIPS (AlexNet)
- FID (torch-fidelity backend)

An interactive HTML dashboard is generated and saved as `metrics_dashboard.html`.

### 2.2 Folder Structure Auto-Compatibility

PSPStain outputs results in:
```
results/<dataset>/<experiment>/val_latest/images/
    ├── real_A
    ├── real_B
    └── fake_B
```

Evaluation is standardized to use:
- Ground truth → `real_B`
- Predictions → `fake_B`

### 2.3 Checkpoint & `val_opt.txt` Mismatch Fix

The `val_opt.txt` shipped with the checkpoint was saved from a different training run and contained incorrect defaults. Always pass these arguments explicitly on the command line to override it:

| Option        | Wrong (`val_opt.txt`) | Correct (launcher) |
|---------------|-----------------------|--------------------|
| `model`       | `cut`                 | `PSPStain`         |
| `netG`        | `resnet_9blocks`      | `resnet_6blocks`   |
| `weight_norm` | `none`                | `spectral`         |
| `ndf`         | `64`                  | `32`               |
| `n_layers_D`  | `3`                   | `5`                |
| `netD`        | `basic`               | `n_layers`         |
| `crop_size`   | `512`                 | `1024`             |
| `load_size`   | `512`                 | `1024`             |

---

## 3. Dataset Mapping & Preparation

PSPStain expects CycleGAN-style aligned datasets:
```
trainA / trainB
testA  / testB
```

### 3.1 BCI Dataset (H&E → IHC)

| Original Path | PSPStain Path       |
|---------------|---------------------|
| HE/train      | datasets/BCI/trainA |
| IHC/train     | datasets/BCI/trainB |
| HE/test       | datasets/BCI/testA  |
| IHC/test      | datasets/BCI/testB  |

> All filenames must match exactly.

### 3.2 MIST Dataset (H&E → Protein Modalities)

Each protein modality becomes its own dataset:
- `MIST_ER`
- `MIST_PR`
- `MIST_HER2`
- `MIST_Ki67`

| Original Path | PSPStain Path |
|---------------|---------------|
| trainA        | trainA        |
| trainB        | trainB        |
| valA          | testA         |
| valB          | testB         |

---

## 4. Training Workflow

### 4.1 BCI Training
```bash
python train.py \
    --dataroot datasets/BCI \
    --name PSPStain_BCI \
    --model PSPStain \
    --netG resnet_6blocks \
    --weight_norm spectral \
    --gpu_ids 0 \
    --batch_size 4 \
    --load_size 1024 \
    --crop_size 512 \
    --unet_seg BCI_unet_seg
```

### 4.2 MIST Training (Example: HER2)
```bash
python train.py \
    --dataroot datasets/MIST_HER2 \
    --name PSPStain_MIST_HER2 \
    --model PSPStain \
    --netG resnet_6blocks \
    --weight_norm spectral \
    --gpu_ids 0 \
    --batch_size 4 \
    --load_size 1024 \
    --crop_size 512 \
    --unet_seg MIST_unet_seg
```

Repeat for: `MIST_ER`, `MIST_PR`, `MIST_Ki67`.

---

## 5. Inference

### 5.1 Single Stain (smoke test — 5 images)
```bash
python test.py \
  --dataroot ~/Downloads/MIST/ER/TrainValAB \
  --name mist \
  --checkpoints_dir checkpoints \
  --results_dir ./results/test_ER \
  --model PSPStain \
  --netG resnet_6blocks \
  --netD n_layers \
  --n_layers_D 5 \
  --ndf 32 \
  --weight_norm spectral \
  --dataset_mode aligned \
  --direction AtoB \
  --phase val \
  --epoch latest \
  --crop_size 1024 \
  --load_size 1024 \
  --num_test 5 \
  --eval \
  --no_flip \
  --gpu_ids 0
```

### 5.2 All MIST Stains (HER2, Ki67, ER, PR)

To run inference on all four MIST subsets at once, use the provided script:
```bash
python run_inference.py
```

> 💡 This processes `MIST_ER`, `MIST_PR`, `MIST_Ki67`, and `MIST_HER2` automatically
> using the single shared checkpoint at `checkpoints/mist/latest_net_G.pth`.

Results are saved to:
```
results/MIST_HER2/mist/val_latest/images/
results/MIST_Ki67/mist/val_latest/images/
results/MIST_ER/mist/val_latest/images/
results/MIST_PR/mist/val_latest/images/
```

### 5.3 Viewing Results
```bash
xdg-open ~/PSPStain/results/MIST_HER2/mist/val_latest/index.html
xdg-open ~/PSPStain/results/MIST_Ki67/mist/val_latest/index.html
xdg-open ~/PSPStain/results/MIST_ER/mist/val_latest/index.html
xdg-open ~/PSPStain/results/MIST_PR/mist/val_latest/index.html
```

---

## 6. Evaluation Workflow

### Run Evaluation
```bash
python eval_metrics.py \
    --gt_dir results/MIST_ER/mist/val_latest/images/real_B \
    --pred_dir results/MIST_ER/mist/val_latest/images/fake_B
```

### Outputs

- `metrics_results.csv` — per-image scores
- `metrics_dashboard.html` — interactive visual dashboard
- Terminal summary
- FID score

---

## Acknowledgement

This repo is built upon
[Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation)
and
[Adaptive Supervised PatchNCE Loss (ASP)](https://github.com/lifangda01/AdaptiveSupervisedPat

