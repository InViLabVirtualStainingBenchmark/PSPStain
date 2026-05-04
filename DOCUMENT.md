# DOCUMENT.md

<!--
This file lives in the root of every forked repo.
Fill it in as you go. Do not reconstruct it after the fact.
Keep entries factual and brief. The audience is a future person
reproducing your setup on a different machine or the HPC cluster.
-->

---

## Model Info

<!--
Copy this information from the upstream repo's README and paper.
"Paired or unpaired" refers to whether the model assumes paired training data.
If the model is domain-specific to virtual staining, note the exact staining task (e.g. H&E to HER2 IHC).
-->

- **Model name:** PSPStain
- **Upstream repo URL:** https://github.com/ccitachi/PSPStain
- **Fork URL:** https://github.com/InViLabVirtualStainingBenchmark/PSPStain
- **Upstream last commit date:** Apr 22, 2026
- **Paper / citation:** - 
- **Paired or unpaired assumption:** paired
- **Intended staining task (if domain-specific):** H&E to IHC (BCI and MIST datasets: HER2, ER, PR, Ki67)

---

## Environment Claimed by Authors

<!--
Record exactly what the authors say in their README or requirements file.
Do not adjust or interpret -- copy their stated versions.
"Requirements file present" should note the filename if it exists.
If no version is specified for Python or PyTorch, write "not specified".
-->

- **Python version:** 3.9
- **PyTorch version:** 1.12.1
- **CUDA version:** 11.6
- **Installation method:** conda
- **Requirements file present:** environment.yml
- **Pretrained weights available:** yes 
- **Pretrained weights notes:**
<!-- Where are they hosted? Are they behind a login? Is the link likely to rot (GDrive, Dropbox, personal server)? -->
- Inference checkpoint: Google Drive (at-risk) and Baidu Drive (at-risk).
Google Drive: https://drive.google.com/file/d/1BCqE_I5ZhLOrESryl2zAd39jU9fc6RvH/view?usp=drive_link
File is named `MIST_net_G.pth` on download; must be renamed and placed at
`checkpoints/mist/latest_net_G.pth` to match the checkpoint name the test script expects.
- UNet segmentation weights (`pretrain/BCI_unet_seg.pth`, `pretrain/MIST_unet_seg.pth`):
bundled in the repo under `pretrain/`. No separate download needed.cond
---

## Environment Actually Used

<!--
Record the environment you actually created and tested in.
If you deviated from what the authors specified, briefly note why (e.g. "authors' version not compatible with CUDA 12.1").
Conda env name should follow the convention: the model's short name.
-->

- **Python version:** 3.9
- **PyTorch version:** 1.13.1+cu117.
- **CUDA version:** 12.1 (driver) cudatoolkit 11.6 bundled via conda
- **Conda environment name:** PSPStain
- **Date tested:** 30.04.2026
- **Hardware:** RTX 4090, WSL2 on Windows 11

---

## Installation

<!--
Follow the authors' README exactly before making any changes.
Record the commands you ran in order.
If an error occurred, paste the key line of the error (not the full traceback) and then record the fix.
If installation succeeded without issues, write "No issues."
-->

### Commands Run

```bash
conda env create -f environment.yml
# After env create, run manually:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install "numpy<2"
```

### Issues and Fixes

<!--
Format: problem encountered -> fix applied.
If no issues, write "None."
-->

| Issue | Fix Applied |
| --- | --- |
| NumPy 2.0.2 installed by default, torchvision 0.13.1 crashes with ARRAY_API error | pip install "numpy<2" -- downgraded to 1.26.4 |
| visdom 0.2.4 missing networkx dependency | Not fixed -- networkx not needed since --display_id 0 disables visdom entirely |
| PyTorch 1.12.1 with cudatoolkit 11.6 does not support RTX 4090 (sm_89) -- nvrtc invalid architecture error | Upgraded to torch==1.13.1+cu117 installed via pip wheel |
| conda solver installs CPU-only PyTorch when given pytorch+nvidia channels | Removed PyTorch from environment.yml entirely; install manually via pip after env create |
| scikit-image, pytorch-msssim, matplotlib missing from environment.yml | Added to pip section of environment.yml |

### GPU Confirmation

<!--
Paste the output of the check below so there is proof the GPU was visible.
Command: python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
-->

```
1.13.1+cu117 True NVIDIA GeForce RTX 4090
```

---

## Dataset Preparation

<!--
Record how the dataset was prepared for this specific model.
"Format expected" means what folder layout or file structure the model's data loader assumes
(e.g. side-by-side paired images, separate A/B folders, CSV manifest, etc.).
"Conversion applied" means any script or command you ran to reformat the standard BCI/MIST-HER2
download into the format this model needs.
If no conversion was needed, write "None -- dataset used as downloaded."
-->

- **Dataset used:** BCI & MIST-HER2
- **Format expected by model:** F2 -- separate trainA/trainB folders for training, valA/valB for inference (phase='val' in test script). testA/testB are NOT used.
- **Conversion applied:** 
    
    ```bash
    # BCI smoke subset (symlinks, no copy)
    ORIG_BCI=~/internship-models/datasets/original/BCI_dataset
    OUT=~/internship-models/datasets/PSPStain-BCI-smoke
    mkdir -p $OUT/trainA $OUT/trainB $OUT/valA $OUT/valB
    ls $ORIG_BCI/HE/train | sort | head -50 | while read f; do ln -sf $ORIG_BCI/HE/train/$f $OUT/trainA/$f; done
    ls $ORIG_BCI/IHC/train | sort | head -50 | while read f; do ln -sf $ORIG_BCI/IHC/train/$f $OUT/trainB/$f; done
    ls $ORIG_BCI/HE/test | sort | head -20 | while read f; do ln -sf $ORIG_BCI/HE/test/$f $OUT/valA/$f; done
    ls $ORIG_BCI/IHC/test | sort | head -20 | while read f; do ln -sf $ORIG_BCI/IHC/test/$f $OUT/valB/$f; done

    # MIST smoke subset (symlinks, no copy)
    ORIG_MIST=~/internship-models/datasets/original/MIST/HER2-004/TrainValAB
    OUT_MIST=~/internship-models/datasets/PSPStain-MIST-smoke
    mkdir -p $OUT_MIST/trainA $OUT_MIST/trainB $OUT_MIST/valA $OUT_MIST/valB
    ls $ORIG_MIST/trainA | sort | head -50 | while read f; do ln -sf $ORIG_MIST/trainA/$f $OUT_MIST/trainA/$f; done
    ls $ORIG_MIST/trainB | sort | head -50 | while read f; do ln -sf $ORIG_MIST/trainB/$f $OUT_MIST/trainB/$f; done
    ls $ORIG_MIST/valA | sort | head -20 | while read f; do ln -sf $ORIG_MIST/valA/$f $OUT_MIST/valA/$f; done
    ls $ORIG_MIST/valB | sort | head -20 | while read f; do ln -sf $ORIG_MIST/valB/$f $OUT_MIST/valB/$f; done
    ```
    
- **Final folder layout used:**
    
    ```
    # sketch the folder tree here, e.g.:
    # PSPStain-BCI-smoke/
    #    trainA/   <-- 50 H&E images (symlinks)
    #    trainB/   <-- 50 IHC images (symlinks)
    #    valA/     <-- 20 H&E images (symlinks)
    #    valB/     <-- 20 IHC images (symlinks)

    #PSPStain-MIST-smoke/
    #    trainA/   <-- 50 H&E images (symlinks)
    #    trainB/   <-- 50 IHC images (symlinks)
    #    valA/     <-- 20 H&E images (symlinks)
    #    valB/     <-- 20 IHC images (symlinks)
    ```
    
- **Number of images used for smoke test (train / test):** 
    - train 50 / test 20

---

## Pretrained Weights

<!--
Only fill this section if pretrained weights exist.
Record the exact download source. Flag any link that is not on a stable host
(Zenodo and HuggingFace are stable; Google Drive, Dropbox, and personal servers are at risk).
Record where you placed the weights relative to the repo root.
-->

- **Download source URL:** https://drive.google.com/file/d/1BCqE_I5ZhLOrESryl2zAd39jU9fc6RvH/view

|Field | Value|
| --- | --- |
|Download source URL | https://drive.google.com/file/d/1BCqE_I5ZhLOrESryl2zAd39jU9fc6RvH/view|
|Host stability	at-risk | (Google Drive)|
|BCI weights placed at | checkpoints/BCI-pretrained/latest_net_G.pth (symlink to checkpoints/mist/BCI_net_G.pth)|
|MIST weights placed at | checkpoints/MIST-pretrained/latest_net_G.pth (symlink to checkpoints/mist/MIST_net_G.pth)|
|Size on disk | 31M each|
|UNet segmentation weights | pretrain/BCI_unet_seg.pth, pretrain/MIST_unet_seg.pth -- bundled in repo, no download needed|

---

## Inference Smoke Test

<!--
Run inference before training if pretrained weights are available -- it is faster
and confirms the code path works independently of the training loop.
Use 10-20 images from the BCI or MIST-HER2 test split.
"Visual check" is a qualitative sanity check only -- not a metric.
Valid outcomes: "images look like expected domain", "blank/grey output", "wrong resolution", "file not written".
-->

- **Script / command run:**
    
    ```bash
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    # BCI
    python test.py --dataroot ~/internship-models/datasets/PSPStain-BCI-smoke --name BCI-pretrained --checkpoints_dir checkpoints --model PSPStain --CUT_mode FastCUT --netG resnet_6blocks --netD n_layers --n_layers_D 5 --ndf 32 --normG instance --normD instance --weight_norm spectral --dataset_mode aligned --direction AtoB --load_size 1024 --crop_size 1024 --phase val --num_test 20 --no_flip --gpu_ids 0
    # MIST-HER2
    python test.py --dataroot ~/internship-models/datasets/PSPStain-MIST-smoke --name MIST-pretrained --checkpoints_dir checkpoints --model PSPStain --CUT_mode FastCUT --netG resnet_6blocks --netD n_layers --n_layers_D 5 --ndf 32 --normG instance --normD instance --weight_norm spectral --dataset_mode aligned --direction AtoB --load_size 1024 --crop_size 1024 --phase val --num_test 20 --no_flip --gpu_ids 0
    ```
    
- **Output folder:** results/BCI-pretrained/val_latest/images/fake_B/
- **Number of output images produced:** 20
- **Output image dimensions:** 1024x1024
- **Visual check result:** IHC-like staining with brown DAB signal on light background, structures spatially aligned with H&E input -- PASS
- **Time to run (approx):**
- **Errors or warnings during inference:** - 
<!-- "None" if clean. Otherwise paste the key error line. -->

---

## Training Smoke Test

<!--
Run training for 5 epochs minimum. The goal is a clean exit, not a useful model.
Use the smallest viable batch size and the model's default resolution unless that causes an OOM error.
Always set checkpoint saving to every epoch (e.g. --save_epoch_freq 1 for pix2pix-style repos)
so there is proof a checkpoint was written.
Monitor GPU memory with: watch -n 1 nvidia-smi (run in a second terminal).
-->

- **Script / command run:**
    
    ```bash
    # BCI
    ## Train
    python train.py --dataroot ~/internship-models/datasets/PSPStain-BCI-smoke --name PSPStain-BCI-scratch --checkpoints_dir checkpoints --model PSPStain --CUT_mode FastCUT --netG resnet_6blocks --netD n_layers --n_layers_D 5 --ndf 32 --normG instance --normD instance --weight_norm spectral --dataset_mode aligned --direction AtoB --load_size 1024 --crop_size 512 --batch_size 1 --n_epochs 1 --n_epochs_decay 0 --save_epoch_freq 1 --display_id 0 --unet_seg BCI_unet_seg --gpu_ids 0

    ## Test
    python test.py --dataroot ~/internship-models/datasets/PSPStain-BCI-smoke --name PSPStain-BCI-scratch --checkpoints_dir checkpoints --model PSPStain --CUT_mode FastCUT --netG resnet_6blocks --netD n_layers --n_layers_D 5 --ndf 32 --normG instance --normD instance --weight_norm spectral --dataset_mode aligned --direction AtoB --load_size 1024 --crop_size 1024 --phase val --num_test 20 --no_flip --gpu_ids 0
    
    ## Evaluate in another terminal with evluation specific environtment (vs-benchmark)
    conda activate vs-benchmark

    python ~/internship-models/evaluate/evaluate.py --pred results/PSPStain-BCI-scratch/val_latest/images/fake_B --gt ~/internship-models/datasets/PSPStain-BCI-smoke/valB --model_name PSPStain --dataset_name BCI --split_name scratch-1epoch-smoke --match_by stem --output ~/internship-models/results.csv

    # MIST-HER2
    ## Train
    python train.py --dataroot ~/internship-models/datasets/PSPStain-MIST-smoke --name PSPStain-MIST-scratch --checkpoints_dir checkpoints --model PSPStain --CUT_mode FastCUT --netG resnet_6blocks --netD n_layers --n_layers_D 5 --ndf 32 --normG instance --normD instance --weight_norm spectral --dataset_mode aligned --direction AtoB --load_size 1024 --crop_size 512 --batch_size 1 --n_epochs 1 --n_epochs_decay 0 --save_epoch_freq 1 --display_id 0 --unet_seg MIST_unet_seg --gpu_ids 0

    ## Test
    python test.py --dataroot ~/internship-models/datasets/PSPStain-MIST-smoke --name PSPStain-MIST-scratch --checkpoints_dir checkpoints --model PSPStain --CUT_mode FastCUT --netG resnet_6blocks --netD n_layers --n_layers_D 5 --ndf 32 --normG instance --normD instance --weight_norm spectral --dataset_mode aligned --direction AtoB --load_size 1024 --crop_size 1024 --phase val --num_test 20 --no_flip --gpu_ids 0

    ## Evaluate in another terminal with evluation specific environtment (vs-benchmark)
    conda activate vs-benchmark

    python ~/internship-models/evaluate/evaluate.py --pred results/PSPStain-MIST-scratch/val_latest/images/fake_B --gt ~/internship-models/datasets/PSPStain-MIST-smoke/valB --model_name PSPStain --dataset_name MIST-HER2 --split_name scratch-1epoch-smoke --match_by stem --output ~/internship-models/results.csv
    ```
    
- **Dataset used:** BCI & MIST-HER2
- **Epochs run:** 1
- **Batch size:** 1
- **Input resolution:** 
- **Time per epoch (approx):**
- **Peak GPU memory (approx, from nvidia-smi):**
- **Checkpoint saved:** yes 
- **Checkpoint path:**
    - /PSPStain/checkpoints/PSPStain-BCI-scratch
    - /PSPStain/checkpoints/PSPStain-MIST-scratch
- **Crash or error during training:**
<!-- "None" if clean. Otherwise paste the key error line and the fix applied. -->

---

## Output Verification

<!--
Open 3-5 output images and compare them visually against the ground-truth target.
This is not a metric -- just a check that the model produced something in the right domain.
"Expected domain" for BCI would be IHC HER2-stained tissue with brown DAB staining on a light background.
Record one or two example output filenames so the check is reproducible.
-->

- **Output folder:** /PSPStain/results
- **Example output filenames:**
- **Dimensions match input:** yes (training uses --crop_size 512 but inference always uses 1024.)
- **Visual sanity check:**
<!-- e.g. "outputs show IHC-like staining, structures roughly aligned with H&E input" -->
- **Any obvious artifacts or failure modes:**

---

## Changes Made to Original Code

<!--
Record every change made to the original repo, no matter how small.
Do not make changes that alter model architecture or training logic.
Only changes needed for the code to run in the benchmark environment are allowed.
Add rows as needed.
-->

| File | Change Description | Reason |
| --- | --- | --- |
| options/test_options.py | Added parser.add_argument('--unet_seg', type=str, default='BCI_unet_seg')
 to TestOptions.initialize() | PSPStainModel reads opt.unet_seg at inference time to load the UNet segmentation weights; original TestOptions did not define this argument, causing an AttributeError |
|  |  |  |

<!--
Common examples of acceptable changes:

- Pinning a dependency version in requirements.txt (e.g. torch==2.1.0) because no version was specified
- Replacing a hardcoded absolute path with a command-line argument
- Removing an import that is not used and is not installable in the benchmark environment
- Adapting the data loader to accept BCI/MIST-HER2 folder structure
-->

---

## Frozen Environment

<!--
After the smoke test passes, export and commit the environment file.
Command: conda env export > environment_<model-name>.yml
This file is what gets adapted for the HPC migration later.
Note any packages that are unusual, very large, or likely to cause conflicts on the cluster.
-->

- **Environment file:** `environment_PSPStain.yml`
- **Committed to fork:** yes
- **Notes on unusual or heavy dependencies:**
<!-- e.g. "requires openslide-python which needs a system-level apt install" -->

---

## HPC Readiness Notes

<!--
Fill this in after the local smoke test passes.
Flag anything that will need attention before running on the VSC cluster.
Common issues: GUI/display dependencies (matplotlib backends), hardcoded CUDA package versions,
dependencies that require apt/system installs, very large model downloads.
Leave blank until local test is complete.
-->

- **Display/GUI dependencies to remove or neutralize:** visdom is imported but not needed. Pass --display_id 0 on all train.py calls.
No code change required -- the flag is sufficient.
- **System-level dependencies (non-pip/conda):** 
    - None. LD_LIBRARY_PATH=/usr/lib/wsl/lib is a WSL2-only workaround and is not
needed on the cluster.
- **Estimated GPU memory requirement:**
- **Estimated storage requirement (weights + data):** 
    - Pretrained weights: ~62 MB total (31 MB x 2).
    - UNet segmentation weights: ~negligible, bundled in repo pretrain/ folder.
    - BCI full dataset: ~3 GB. MIST-HER2 full dataset: ~3 GB.
    - Allow ~10 GB total on $VSC_SCRATCH per dataset during training.
- **Other notes for cluster adaptation:**
    - CRITICAL: The cluster module stack provides PyTorch 2.1.2. PSPStain requires
    - PyTorch 1.13.1+cu117. The system-site-packages venv approach used for other models cannot be used here -- the system PyTorch version is incompatible.
    - A fully self-contained venv (without --system-site-packages) is required, with PyTorch 1.13.1+cu117 installed via pip wheel inside the venv. All pip dependencies must be installed inside the venv:
    torch==1.13.1+cu117, torchvision==0.14.1+cu117 (via [pytorch.org](http://pytorch.org/) wheel index),
    numpy<2, opencv-python==4.8.1.78, scikit-image, pytorch-msssim, matplotlib, visdom, GPUtil, dominate, scipy, packaging.
    - The pretrain/ folder (BCI_unet_seg.pth, MIST_unet_seg.pth) must be transferred to the cluster alongside the repo -- it is bundled in the repo root and is not downloaded at runtime.
    - PSPStain does not fetch any weights or assets at runtime. Compute nodes having no outbound internet access is not a problem for this model.
 
---

## Summary

<!--
Write 2-4 sentences summarizing what worked, what did not, and what the next step is.
Be specific. Include the overall pass/fail verdict.
This is the first thing someone reads when picking this model back up.
-->

**Overall result:** PASS

<!-- Example pass:
"[Model] smoke test completed on [date]. Inference with pretrained weights passed on 10 BCI test images.
Training ran for 5 epochs without crash. One change was made to the data loader to accept separate
source/target folders. Frozen environment committed. Ready for full benchmark run."

Example fail:
"[Model] smoke test failed at the environment step. The required PyTorch version (1.4) is not
compatible with CUDA 12.1. Blocked until a workaround is identified. Do not schedule for HPC."
-->