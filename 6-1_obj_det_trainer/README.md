# 6-1 · Object Detection Trainer

**Stage 3 · Train and test** · **Type: Training** · [← Back to MatchPlant](../README.md)

Train a Faster R-CNN model (ResNet-50 FPN v2 backbone) from the start on the tiled COCO datasets produced by [5 Image Splitter](../5_img_splitter/README.md). It runs on an NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU, on a workstation, cloud, or HPC system.

This is one of two training routes. Use **6-1** to train on a new dataset; use [6-2 Transfer Learning](../6-2_obj_det_trans_learner/README.md) to fine-tune an existing model instead.

| | |
| --- | --- |
| **Input** | `data/train`, `data/val`, `annotations/train.json`, `annotations/val.json` from [5 Image Splitter](../5_img_splitter/README.md) |
| **Output** | `checkpoints/best_model.pt` and per-epoch validation results |
| **Previous step** | [5 Image Splitter](../5_img_splitter/README.md) |
| **Next step** | [7 Object Detection Tester](../7_obj_det_tester/README.md): evaluate `best_model.pt` |

## Run it from the dashboard

1. On the dashboard home, select **Object Detection Trainer**.
2. If the page reports missing packages, click **Install requirements** and wait for it to finish. For an NVIDIA GPU, see [GPU setup](#gpu-setup).
3. Click **Browse…** to choose:
   - **Training images folder** (`data/train`) and **Training annotations file** (`annotations/train.json`)
   - **Validation images folder** (`data/val`) and **Validation annotations file** (`annotations/val.json`)
4. Choose where to save results (optional):
   - **Experiment name**, for example `maize_test_01`, saves to `runs/maize_test_01/` in this module folder; or
   - **Save results to folder** to pick any folder (this takes priority over the experiment name).
5. Review the training settings (defaults below). Help icons explain each field.
6. Click **Run** and follow the training log in the **Output** panel. Training can take hours; **Stop** cancels it.
7. When it finishes, use `checkpoints/best_model.pt` in [7 Object Detection Tester](../7_obj_det_tester/README.md).

Use a new experiment name or folder for each run so results are not mixed. Relative paths are read from this module's folder.

## Settings

| Field | Default | Notes |
| --- | --- | --- |
| Training epochs | 150 | |
| Batch size | 1 | Increase only if GPU memory allows |
| Learning rate | 0.05 | |
| Validation interval (epochs) | 1 | Validate every N epochs; 5 saves time on long runs |
| Best model selection metric | `ap50` | `ap50` (AP at IoU 0.50) or `map50_95` (COCO mAP at IoU 0.50:0.95) |
| Random seed (optional) | blank | Set a number (for example 42) for reproducible runs |
| Early stopping patience (optional) | blank | Stop after N validations without improvement; blank runs all epochs |
| Data loading workers (optional) | automatic | Chosen from RAM/CPU on Windows/Linux; 0 on macOS |
| Disable mixed precision | off | Mixed precision is used on CUDA GPUs unless disabled |
| Disable pinned memory | off | |
| Clear GPU cache after each step | off | Only for memory problems; slower |

## Outputs

```text
runs/maize_test_01/                      # or your chosen folder
├── checkpoints/
│   ├── best_model.pt                    # → 7 Object Detection Tester
│   ├── model_epoch_*.pt
│   └── emergency_save.pt                # only if training stops on an error
└── validation_results/
    ├── validation_results_epoch_*.json
    └── training_summary.jsonl
```

If you leave both the experiment name and the folder blank, results go to `checkpoints/` and `validation_results/` in this module folder.

Validation metrics are named `AP_50` (AP at IoU 0.50) and `mAP_50_95` (COCO mAP at IoU 0.50:0.95). The first valid best checkpoint is saved even if its score is zero. Training stops on failed batches, non-finite losses, or evaluation errors.

## GPU setup

Modules 6-1, 6-2, and 7 use PyTorch 2.6.0 with torchvision 0.21.0. For an NVIDIA GPU, install the CUDA build of that pair from the [PyTorch selector](https://pytorch.org/get-started/locally/) into this module's environment (or point the module to an existing PyTorch/CUDA environment on the dashboard **Settings** page).

## Run without the dashboard

From this folder:

```bash
pip install -r requirements.txt
python train.py \
  --output_dir runs/my_training_run \
  --train-data-dir data/train \
  --val-data-dir data/val \
  --train-annotation-file annotations/train.json \
  --val-annotation-file annotations/val.json
```

`python train.py` with no arguments uses the default paths `data/train`, `data/val`, `annotations/train.json`, and `annotations/val.json`.

**Cloud/HPC:** assign one job per GPU, and start with a modest worker count:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir runs/my_training_run \
  --batch-size 1 --num-workers 4 --val-frequency 5
```

If a GPU is underused, running independent jobs (for example, different seeds) on separate GPUs is usually faster than spreading one model across GPUs.

<details>
<summary>All command-line options</summary>

```text
--run_name                   Experiment name; saves under runs/<run_name> unless --output_dir is set
--output_dir                 Folder for checkpoints and validation results
--train-data-dir / --val-data-dir
--train-annotation-file / --val-annotation-file
--seed                       Optional reproducibility seed (Python, NumPy, PyTorch, data shuffling, augmentation)
--epochs                     Number of epochs, default 150
--batch-size                 Images per batch, default 1
--lr                         Learning rate, default 0.05
--num-workers                DataLoader workers
--val-frequency              Validate every N epochs, default 1
--early-stopping-patience    Stop after N validations without improvement
--best-metric ap50|map50_95  Metric for selecting best_model.pt
--no-amp                     Disable CUDA mixed precision
--no-pin-memory              Disable pinned memory
--clear-cache-each-step      Clear GPU cache every step (slower)
```

Run `python train.py --help` for the full list.

</details>

## Troubleshooting

- **Out of GPU memory:** keep batch size 1, enable **Clear GPU cache after each step**, or use smaller tiles in Image Splitter.
- **Very slow:** check that the log reports a GPU (CUDA or MPS) rather than CPU, and see [GPU setup](#gpu-setup).
- **Annotation errors:** use the matching image folder and annotation file from the same Image Splitter output.

## License

MIT License. See [LICENSE](../LICENSE).
