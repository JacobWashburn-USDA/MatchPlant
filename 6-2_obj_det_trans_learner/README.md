# 6-2 · Transfer Learning

**Stage 3 · Train and test** · **Type: Training** · [← Back to MatchPlant](../README.md)

Fine-tune a pretrained Faster R-CNN (ResNet-50 FPN) checkpoint on a new dataset. Transfer learning reuses what the model already learned, so it usually needs less data and less training time than training from the start.

This is one of two training routes. Use **6-2** to adapt an existing model; use [6-1 Object Detection Trainer](../6-1_obj_det_trainer/README.md) to train a new model instead.

| | |
| --- | --- |
| **Input** | A pretrained checkpoint (`.pt`), plus `data/train`, `data/val`, `annotations/train.json`, `annotations/val.json` from [5 Image Splitter](../5_img_splitter/README.md) |
| **Output** | `best_model_transfer.pt` and per-epoch validation reports |
| **Previous step** | [5 Image Splitter](../5_img_splitter/README.md) |
| **Next step** | [7 Object Detection Tester](../7_obj_det_tester/README.md): evaluate `best_model_transfer.pt` |

**Need a pretrained model?** The MatchPlant maize model is in the [dataset on Zenodo](../README.md#download-dataset) (`Pre-trained model_Faster R-CNN.pt`). A checkpoint from [6-1](../6-1_obj_det_trainer/README.md) also works.

## Run it from the dashboard

1. On the dashboard home, select **Transfer Learning**.
2. If the page reports missing packages, click **Install requirements** and wait for it to finish. For an NVIDIA GPU, see [GPU setup](../6-1_obj_det_trainer/README.md#gpu-setup).
3. Click **Browse…** to choose:
   - **Pretrained model file** (`.pt`)
   - **Training images folder** and **Training annotations file**
   - **Validation images folder** and **Validation annotations file**
   - **Save checkpoints to folder** and **Save validation reports to folder**: use a new folder for each run, for example `runs/transfer_01/checkpoints`
4. Review the training settings (defaults below).
5. Click **Run**. The form is checked and saved to `transfer_config.yaml`, then training starts. Follow the log in the **Output** panel.
6. When it finishes, use `best_model_transfer.pt` in [7 Object Detection Tester](../7_obj_det_tester/README.md).

**Save settings** saves the form without running. Less common model options are under **Advanced YAML settings**; save them with **Save YAML settings**.

## Settings

| Field | Default | Notes |
| --- | --- | --- |
| Training epochs | 50 | |
| Batch size | 1 | Lower it if you run out of memory |
| Starting epoch | 0 | Only changes epoch numbering; it does not resume an earlier run. Keep 0 for a new run. |
| Freeze backbone | on | The feature extractor is not trained (recommended for small datasets) |
| Freeze region proposal network | off | |
| Freeze detection heads | off | |
| Backbone learning rate | 0.0001 | Used only if the backbone is not frozen |
| Region proposal learning rate | 0.001 | |
| Detection heads learning rate | 0.001 | |

At least one part of the model must stay trainable. See the [Transfer Learning Guide](transfer_learning_guide.md) for freezing strategies and examples.

The learning rates drop to one tenth every 10 epochs (SGD, momentum 0.9, weight decay 0.0005, gradient clipping 1.0).

## Checkpoint requirements

- A trusted Faster R-CNN ResNet-50 FPN (v1 or v2) checkpoint, saved as a state dictionary or with a `model_state_dict` / `state_dict` key.
- The checkpoint's number of classes (including background) must match your dataset. This module fine-tunes the existing classifier; it does not change the number of classes.
- COCO category IDs must start at 1 (0 is background) and match the model's labels.
- The model version (v1/v2) and MatchPlant anchor layouts are detected automatically. For a custom anchor layout, set `anchor_sizes`, `anchor_ratios`, and `anchor_source: config` under **Advanced YAML settings**.

If the checkpoint does not match, training stops with an error instead of silently using random weights.

## Outputs

```text
runs/transfer_01/
├── checkpoints/
│   ├── best_model_transfer.pt           # best AP50 → 7 Object Detection Tester
│   ├── model_transfer_epoch_N.pt        # periodic checkpoints
│   └── emergency_save_transfer.pt       # only if training fails
└── validation_results/
    ├── per-epoch JSON reports
    └── training_summary.jsonl
```

If the output folders are left at their defaults, results go to `checkpoints/` and `validation_results/` in this module folder. If `best_model_transfer.pt` already exists in the chosen folder, training will not start, so earlier results are never overwritten.

## Run without the dashboard

Keep the full MatchPlant folder: this module uses code from `../6-1_obj_det_trainer/train.py`.

```bash
pip install -r requirements.txt
python transfer_train.py --config transfer_config.yaml --validate-only   # check settings and paths
python transfer_train.py --config transfer_config.yaml                   # train
```

Edit paths and settings in [transfer_config.yaml](transfer_config.yaml). Relative paths are read from the YAML file's folder. Output folders can also be set there:

```yaml
training:
  checkpoint_dir: runs/transfer_01/checkpoints
  validation_dir: runs/transfer_01/validation_results
  save_freq: 5
```

## Troubleshooting

- **Checkpoint shape or class mismatch:** the checkpoint was trained with a different number of classes or anchors; see [Checkpoint requirements](#checkpoint-requirements).
- **"best_model_transfer.pt already exists":** choose a new checkpoint folder.
- **Out of memory:** lower the batch size, or keep the backbone frozen.
- **Very slow:** check that the log reports a GPU (CUDA or MPS) rather than CPU.

<details>
<summary>For developers</summary>

```bash
python -m unittest discover -s tests
python tests/smoke_transfer.py
```

The smoke test trains one CPU epoch on temporary synthetic data and checks the saved output. It does not measure model quality.

</details>

## License

MIT License. See [LICENSE](../LICENSE).
