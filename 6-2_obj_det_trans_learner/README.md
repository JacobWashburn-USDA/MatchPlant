# Object Detection Transfer Learning — Module 6.2

Fine-tune an existing Faster R-CNN ResNet50-FPN checkpoint on new images with COCO annotations. Module 6.2 is an alternative to training in Module 6.1.

## Using the dashboard

Open **Transfer Learning** in the [local dashboard](../dashboard/README.md). Use **Browse…** to select your pretrained checkpoint, training/validation image folders, and their COCO JSON files. Set epochs, batch size, freezing options, and learning rates. Help icons explain the settings.

**Save settings** saves the form to `transfer_config.yaml`. **Run** validates and saves the form before starting training. **Advanced YAML settings** contains model and output options. Save advanced edits using **Save YAML settings**. Updating existing scalar fields preserves comments; adding missing fields can rewrite YAML and remove comments.

Defaults: 50 epochs, batch size 1, starting epoch 0, backbone frozen, RPN and ROI heads trainable. Learning rates are 0.0001 for the backbone and 0.001 for each head. At least one component must remain trainable.

## Quick start

Keep the complete MatchPlant folder structure: the script loads dataset, transforms, device selection, and COCO evaluation directly from `../6-1_obj_det_trainer/train.py`. No supporting scripts need to be copied into this folder.

Use a separate Python environment. Install a matching PyTorch/torchvision pair for your hardware using the [official installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```sh
python -m pip install -r requirements.txt
```

The Windows CPU integration test used Python 3.11, torch 2.6.0, and torchvision 0.21.0. That CPU pair can be installed with:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

For GPU training, use the appropriate CUDA build from the selector. Select that environment in the dashboard's module interpreter settings.

Choose your checkpoint and data paths through the dashboard or edit `transfer_config.yaml`. Relative paths are resolved against the YAML file's folder, including when using a custom config:

```sh
python transfer_train.py --config transfer_config.yaml --validate-only
python transfer_train.py --config transfer_config.yaml
```

Validation-only checks settings and path existence; training additionally checks checkpoint compatibility and reads the datasets.

## Input requirements

- A trusted Faster R-CNN ResNet50-FPN v1 or v2 checkpoint, stored as a state dictionary or with `model_state_dict` / `state_dict`. The repository [dataset downloads](../README.md#download-dataset) include a pretrained model.
- Matching `model.num_classes`, including the background class. This script fine-tunes the existing head; it does not replace the classifier for a different number of classes.
- Training and validation image folders, each with a corresponding COCO annotation JSON. Both splits need annotated images. Category IDs must match model labels, starting at 1; background is 0.
- The sibling Module 6.1 folder and the packages in `requirements.txt`.

## Checkpoint compatibility

`model.architecture: auto` detects v1 versus v2 from the checkpoint's box head. `model.anchor_source: auto` recognizes MatchPlant's default 3-anchor, custom 12-anchor, and legacy 40-anchor layouts when the configured anchor count differs. One shared anchor group is expanded across all five FPN levels. For custom layouts, set matching `anchor_sizes` and `anchor_ratios` and `anchor_source: config`. Anchor sizes themselves are not stored in model weights; custom checkpoints need their original configuration.

Checkpoint shape mismatches, missing weights, and unexpected weights produce an error instead of silently leaving random model parameters. Weights are loaded with PyTorch's restricted `weights_only=True` loader. Unsupported checkpoint objects require a trusted weights-only export.

## Outputs

By default, checkpoints go into `checkpoints/` and validation reports into `validation_results/`, beside the YAML file. Change them in **Advanced YAML settings**:

```yaml
training:
  checkpoint_dir: runs/transfer_01/checkpoints
  validation_dir: runs/transfer_01/validation_results
  save_freq: 5
```

The best AP50 checkpoint is `best_model_transfer.pt`; the first valid epoch is saved even when AP50 is zero. Periodic checkpoints are `model_transfer_epoch_N.pt`, with zero-based epoch numbers. Checkpoints include model, optimizer, scheduler, metrics, and the resolved model configuration. Validation produces per-epoch JSON reports and a `training_summary.jsonl` log.

Use the form’s checkpoint and validation folder fields to choose output locations. A typed new folder is created automatically. Choose a new checkpoint folder for a new run. An existing `best_model_transfer.pt` blocks training to avoid overwriting it. A failed training/evaluation attempt writes `emergency_save_transfer.pt` where possible and reports failure to the dashboard.

**Starting epoch** changes the epoch numbering and final epoch range; it does not resume optimizer or scheduler state. Start at 0 for a new fine-tuning run.

## Training behavior

The script uses separate SGD learning-rate groups, momentum 0.9, weight decay 0.0005, gradient clipping at 1.0, and a StepLR schedule that multiplies learning rates by 0.1 every 10 completed epochs. A frozen backbone stays in evaluation mode during training. Frozen RPN/ROI parameters remain in the training loss path. Training errors and non-finite losses stop the run.

CPU, CUDA, and macOS MPS device selection comes from Module 6.1. CPU runs are slower; reduce batch size if memory is limited. Native macOS and CUDA execution still need testing on those devices. See the [Transfer Learning Guide](transfer_learning_guide.md) for freezing strategies and configuration examples.

## Checks for developers

```sh
python -m unittest discover -s tests
python tests/smoke_transfer.py
```

The smoke test creates temporary synthetic images and a checkpoint, trains one real CPU epoch, evaluates COCO metrics, and checks the saved output. It does not measure model quality on real data.
