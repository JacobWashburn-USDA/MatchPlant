# 7 · Object Detection Tester

**Stage 3 · Train and test** · **Type: Form** · [← Back to MatchPlant](../README.md)

Evaluate a trained checkpoint on the held-out test split and report detection metrics. It also saves the model's predictions, which [8 Image-to-Ortho Projector](../8_img_to_ortho/README.md) places on the orthomosaic.

| | |
| --- | --- |
| **Input** | `data/test` and `annotations/test.json` from [5 Image Splitter](../5_img_splitter/README.md), and a checkpoint from [6-1](../6-1_obj_det_trainer/README.md) or [6-2](../6-2_obj_det_trans_learner/README.md) |
| **Output** | Metrics (JSON/CSV), `coco_predictions.json`, and plots |
| **Previous step** | [6-1 Object Detection Trainer](../6-1_obj_det_trainer/README.md) or [6-2 Transfer Learning](../6-2_obj_det_trans_learner/README.md) |
| **Next step** | [8 Image-to-Ortho Projector](../8_img_to_ortho/README.md): use `coco_predictions.json` |

## Run it from the dashboard

1. On the dashboard home, select **Object Detection Tester**.
2. If the page reports missing packages, click **Install requirements** and wait for it to finish.
3. Click **Browse…** to choose:
   - **Data dir**: the test images folder (`data/test`)
   - **Annotation file**: the test annotations (`annotations/test.json`)
   - **Checkpoint**: `best_model.pt` (6-1) or `best_model_transfer.pt` (6-2). Browse to the actual file; it is usually in the training module's folder.
   - **Results dir**: where to save results, for example `runs/my_training_run/test_results` (a new folder is created)
4. Click **Run** and follow the **Output** panel.
5. Check the results, then use `coco_predictions.json` in [8 Image-to-Ortho Projector](../8_img_to_ortho/README.md).

Relative paths are read from this module's folder.

## Outputs

```text
results_dir/
├── test_summary.json            # main metrics
├── test_summary.csv             # same metrics as a table
├── paper_metrics.json
├── coco_predictions.json        # predictions → 8 Image-to-Ortho Projector
├── confidence_distribution.png
└── confusion_matrix.png
```

Metrics reported:

- `mAP_50_95`: COCO mAP at IoU 0.50:0.95
- `AP_50`: AP at IoU 0.50
- Precision, recall, and F1 at IoU 0.5

## Run without the dashboard

Keep the full MatchPlant folder: this module uses code from `../6-1_obj_det_trainer/train.py` (and reads settings saved by 6-2).

```bash
pip install -r requirements.txt
python test.py \
  --data-dir data/test \
  --annotation-file annotations/test.json \
  --checkpoint ../6-1_obj_det_trainer/runs/my_training_run/checkpoints/best_model.pt \
  --results-dir ../6-1_obj_det_trainer/runs/my_training_run/test_results
```

Without arguments, `python test.py` uses `data/test`, `annotations/test.json`, `checkpoints/best_model.pt`, and `paper_results/`. On a multi-GPU system, add `CUDA_VISIBLE_DEVICES=0` before the command to pick a GPU.

## Tips

- Test each trained model once on the same fixed test set.
- Running the test again on the same checkpoint does not measure model variability. To measure it, train several models with different seeds in [6-1](../6-1_obj_det_trainer/README.md) and test each one.

## Troubleshooting

- **Checkpoint not compatible:** the tester supports 6-1 and 6-2 checkpoints and detects v1/v2 models and MatchPlant anchor layouts automatically. Other checkpoints are rejected.
- **Checkpoint not found:** browse to the file instead of typing a path relative to another module.
- **Annotation errors:** use the test images and `test.json` from the same Image Splitter output.

<details>
<summary>For developers</summary>

```bash
python -m unittest discover -s tests
python tests/smoke_training_handoff.py
```

The smoke check trains a small CPU model in 6-1, fine-tunes it in 6-2, and tests it here on temporary synthetic data. It checks that the steps run, not model accuracy.

</details>

## License

MIT License. See [LICENSE](../LICENSE).
