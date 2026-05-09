# Object Detection Training Tool

This module trains a Faster R-CNN object detector on COCO-format tiled imagery. It is designed for normal workstation, cloud, or HPC GPU use while keeping simple defaults for local runs.

## Features

- Faster R-CNN with ResNet-50 FPN v2 backbone
- COCO-format training and validation datasets
- CUDA, Apple MPS, or CPU device selection
- Optional reproducible seed control
- Run-specific output folders
- Configurable batch size, DataLoader workers, validation frequency, and best-checkpoint metric
- COCO validation metrics with clear metric names:
  - `mAP_50_95` = COCO mAP at IoU 0.50:0.95
  - `AP_50` = AP at IoU 0.50

## Requirements

Install the module dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include PyTorch, torchvision, Pillow, NumPy, psutil, and pycocotools.

## Input Structure

Expected default structure:

```text
project_root/
  train.py
  data/
    train/
    val/
  annotations/
    train.json
    val.json
```

The `data/` and `annotations/` folders are normally produced by `5_img_splitter`.

## Quick Start

From this module folder:

```bash
python train.py
```

This uses the default paths:

```text
data/train
annotations/train.json
data/val
annotations/val.json
checkpoints/
validation_results/
```

## Recommended GPU/Cloud Run

Use an output directory so each training run is self-contained:

```bash
python train.py   --output_dir runs/my_training_run   --train-data-dir data/train   --val-data-dir data/val   --train-annotation-file annotations/train.json   --val-annotation-file annotations/val.json   --batch-size 1   --num-workers 4   --val-frequency 5   --best-metric ap50
```

To run on a specific GPU on Linux/HPC:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py   --output_dir runs/my_training_run   --train-data-dir data/train   --val-data-dir data/val   --train-annotation-file annotations/train.json   --val-annotation-file annotations/val.json   --batch-size 1   --num-workers 4   --val-frequency 5
```

## Optional Reproducible Run

A seed is optional for normal use, but useful when you need reproducibility:

```bash
python train.py   --seed 42   --output_dir runs/seed_42   --train-data-dir data/train   --val-data-dir data/val   --train-annotation-file annotations/train.json   --val-annotation-file annotations/val.json
```

The seed controls Python, NumPy, PyTorch CPU/CUDA randomness, DataLoader shuffling, and augmentation randomness.

## Useful Arguments

```text
--output_dir                 Folder for checkpoints and validation logs
--seed                       Optional reproducibility seed
--epochs                     Number of epochs, default 150
--batch-size                 Images per batch, default 1
--num-workers                DataLoader workers
--val-frequency              Validate every N epochs, default 1
--best-metric ap50|map50_95  Select best checkpoint by AP@0.5 or mAP@0.5:0.95
--no-amp                     Disable CUDA mixed precision
--clear-cache-each-step      Debug option for memory pressure; slower
```

## Outputs

When `--output_dir runs/my_training_run` is used:

```text
runs/my_training_run/
  checkpoints/
    best_model.pt
    model_epoch_*.pt
    emergency_save.pt
  validation_results/
    validation_results_epoch_*.json
    training_summary.jsonl
```

If `--output_dir` is not provided, outputs are saved to the legacy default folders:

```text
checkpoints/
validation_results/
```

## Notes

- `AP_50` and `mAP_50_95` are intentionally named separately to avoid confusion with COCOeval indexing.
- For large A100/HPC jobs, start with `--batch-size 1 --num-workers 4 --val-frequency 5` and increase workers only if the filesystem can handle it.
- If the GPU is underutilized, the best speedup is often running independent training jobs on separate GPUs rather than spreading one model over multiple GPUs.

## License

This project is licensed under the MIT License. See the repository `LICENSE` file.
