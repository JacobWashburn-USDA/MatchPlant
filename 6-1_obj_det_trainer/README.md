# Object Detection Training Tool

This module trains a Faster R-CNN object detector on COCO-format tiled imagery. It is designed for normal workstation, cloud, or HPC GPU use while keeping simple defaults for local runs.


## Using the dashboard

Open **Object Detection Trainer** in the [local dashboard](../dashboard/README.md). Install requirements if prompted. Use **Browse…** to select training/validation image folders and COCO annotation files. Help icons explain the fields; defaults remain visible. Click **Run** to follow the live training log.

- **Experiment name (optional)** corresponds to `--run_name`. For example, `maize_test_01` saves to `runs/maize_test_01/` inside this module.
- **Save results to folder (optional)** corresponds to `--output_dir`. It takes priority over the experiment name. Browse for an existing folder, or type a new path.
- With both fields blank, checkpoints go to `checkpoints/` and validation results to `validation_results/` inside this module. Choose a distinct experiment name or output folder for each experiment.
- Relative paths are resolved from this module's folder, not the dashboard folder.

Current defaults: 150 epochs, batch size 1, learning rate 0.05, validation every epoch, and best metric `ap50`. The random seed is unset and early stopping is disabled. Data loading workers are chosen automatically from RAM/CPU on Windows/Linux and are 0 on macOS. CUDA mixed precision and pinned memory are enabled when applicable unless disabled; clearing the GPU cache after each step is off.

Training uses torchvision’s default five-level FPN anchors, matching existing Module 6.1 checkpoints. Modules 6.1, 6.2, and 7 use the same pinned PyTorch/torchvision pair (2.6.0/0.21.0). For CUDA, install the corresponding build from the [official PyTorch selector](https://pytorch.org/get-started/locally/). Runtime requirements omit optional notebook tools. Training stops on failed batches, non-finite losses, or evaluation errors, and saves the first valid best checkpoint even when the validation score is zero.

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
--run_name                   Experiment name; saves under runs/<run_name> unless output_dir is set
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

If `--output_dir` is blank but `--run_name` is supplied, outputs are saved to `runs/<run_name>/checkpoints/` and `runs/<run_name>/validation_results/`. Only when both are blank are outputs saved to the default folders:

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
