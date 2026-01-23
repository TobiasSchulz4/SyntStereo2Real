# SyntStereo2Real (Reproduction)

This repository provides a lightweight reproduction of **SyntStereo2Real: Edge-Aware GAN for Remote Sensing Image-to-Image Translation while Maintaining Stereo Constraint**. It implements the edge-aware generator, PatchGAN discriminators, Sobel edge extraction, disparity-based warping loss, and a wrapper for downstream AANet training/evaluation.

## 1. Overview

Core features implemented:
- Sobel edge extraction (GPU conv) and edge encoder injection
- MUNIT-style content encoder + AdaIN decoder with fixed per-domain styles
- PatchGAN discriminators
- Differentiable warping loss: L1 + (1 - SSIM)
- Full translation training loop with reconstruction, cycle, adversarial, and warping losses
- Dataset translation utility
- Downstream AANet training and evaluation wrappers

Project layout (important folders):
```
configs/                   # training configs
src/                       # models, losses, dataloaders, trainers
experiments/run_scripts/   # end-to-end run scripts
third_party/aanet/         # AANet repo placeholder
```

## 2. Environment Setup

Python 3.8–3.10 recommended. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: Install the appropriate CUDA-enabled PyTorch wheel that matches your system. See [PyTorch install](https://pytorch.org/get-started/locally/).

## 3. Dataset Layout

Prepare datasets into the following layout (produced by `data/scripts/*`):

**Synthetic stereo (e.g., SyntCities or SceneFlow/Driving)**
```
datasets/<synthetic_name>/
  left/
  right/
  disp/
  files.txt   # optional list of stems
```

**Real images (e.g., US3D or KITTI)**
```
datasets/<real_name>/
  images/
  files.txt   # optional list of stems
```

### Preparation scripts
Examples:
```bash
python -m data.scripts.prepare_syntcities --src_root /path/to/raw/syntcities --out_root datasets/syntcities
python -m data.scripts.prepare_us3d --src_root /path/to/raw/us3d --out_root datasets/us3d
python -m data.scripts.prepare_scene_flow --src_root /path/to/raw/scene_flow --out_root datasets/scene_flow
python -m data.scripts.prepare_kitti --src_root /path/to/raw/kitti --out_root datasets/kitti2015
```

## 4. Translation Training

Configs:
- `configs/config_translation_syntcities_us3d.yaml`
- `configs/config_translation_driving_kitti.yaml`

Run translation training:
```bash
python -m src.trainers.train_translation \
  --config configs/config_translation_syntcities_us3d.yaml \
  --output_root experiments/checkpoints/syntcities_to_us3d \
  --device cuda
```

Training uses:
- batch_size = 4
- epochs = 100
- LR decay after epoch 50
- λ_rec=0.8, λ_cycle=10, λ_adv=10, λ_warp_l1=1, λ_warp_ssim=1

## 5. Translate Synthetic Dataset

Translate the full synthetic dataset to the real domain:
```bash
python -m src.trainers.translate_dataset \
  --config configs/config_translation_syntcities_us3d.yaml \
  --checkpoint experiments/checkpoints/syntcities_to_us3d/checkpoints/checkpoint_0001000.pt \
  --output_root datasets/translated/syntcities_to_us3d \
  --device cuda \
  --use_edge
```

Output structure:
```
datasets/translated/syntcities_to_us3d/
  left/
  right/
  disp/   # original disparity maps (saved as .npy)
```

## 6. Downstream AANet Training

AANet is expected in `third_party/aanet` (submodule or local copy). This repo provides a wrapper to call its training script.

Train AANet on translated data:
```bash
python -m src.trainers.train_aanet --config configs/config_aanet_training.yaml
```

Update `configs/config_aanet_training.yaml` to point to your translated dataset path.

## 7. Evaluation

Evaluate predictions vs ground truth (MAD, 1px/3px accuracy):
```bash
python -m src.trainers.eval_aanet --root datasets/aanet_eval/us3d
```

Expected directory structure for evaluation:
```
root/
  pred/
  gt/
  mask/   # optional
```

## 8. Quick Inference

Run a single-image translation:
```bash
python -m src.scripts.quick_inference \
  --checkpoint experiments/checkpoints/syntcities_to_us3d/checkpoints/checkpoint_0001000.pt \
  --input_left path/to/image.png \
  --output_left out.png \
  --use_edge
```

## 9. End-to-End Scripts

Convenient pipeline scripts are in `experiments/run_scripts/`:
```bash
bash experiments/run_scripts/run_syntcities_to_us3d.sh
bash experiments/run_scripts/run_driving_to_kitti.sh
```

## 10. Notes & Tips

- Disparity convention: warping uses `x_s = x + d`. If your dataset uses opposite sign, invert disparity before training.
- SSIM expects inputs in [0,1]. The training code maps tensors from [-1,1] to [0,1] before computing SSIM.
- For deterministic runs, seeds are saved in checkpoints and `torch.backends.cudnn.deterministic=True` is set in training.

## 11. License / Disclaimer

This is a research reproduction scaffold intended for experimentation. You must obtain datasets and third-party code (AANet) yourself and comply with their licenses.
