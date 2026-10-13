# Object Detection Testing Tool

This module evaluates a trained Faster R-CNN checkpoint on a fixed COCO-format test set. It is intended to run once per trained model and save machine-readable summaries for reporting or downstream projection.


## Using the dashboard

Open **Object Detection Tester** in the [local dashboard](../dashboard/README.md). Install requirements if prompted. Browse for the held-out test images folder, its COCO annotations, and the model checkpoint. Select a results folder or type a new output path. Click **Run** and follow the live log.

Default paths are `data/test`, `annotations/test.json`, `checkpoints/best_model.pt`, and `paper_results`, relative to this module's folder. A checkpoint from Module 6.1 usually lives elsewhere; browse to the actual file rather than assuming its relative path is shared between modules.

The tester accepts compatible checkpoints from **Module 6.1 or Module 6.2**, detects v1/v2 architectures and known MatchPlant anchor layouts, and honors `model_config` saved by transfer training. Keep both sibling modules beside Module 7. It rejects incompatible weights. Plots use a display-independent backend, output folders can be created recursively, and the confusion plot aggregates all test images. Size-based counts match each ground-truth object at most once.

## Features

- Loads a trained checkpoint from Module 6.1 or 6.2
- Evaluates on a fixed held-out test set
- Computes COCO metrics:
  - `mAP_50_95` = COCO mAP at IoU 0.50:0.95
  - `AP_50` = AP at IoU 0.50
- Computes precision, recall, and F1 at IoU 0.5
- Saves prediction output in COCO-style format
- Saves JSON and CSV summaries

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include PyTorch, torchvision, Pillow, NumPy, matplotlib, and pycocotools.

## Input Structure

Expected default structure:

```text
project_root/
  test.py
  data/
    test/
  annotations/
    test.json
  checkpoints/
    best_model.pt
```

The trained checkpoint normally comes from `6-1_obj_det_trainer`.

## Quick Start

From this module folder:

```bash
python test.py
```

This uses:

```text
data/test
annotations/test.json
checkpoints/best_model.pt
paper_results/
```

## Recommended Run

```bash
python test.py   --data-dir data/test   --annotation-file annotations/test.json   --checkpoint runs/my_training_run/checkpoints/best_model.pt   --results-dir runs/my_training_run/test_results
```

To run on a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py   --data-dir data/test   --annotation-file annotations/test.json   --checkpoint runs/my_training_run/checkpoints/best_model.pt   --results-dir runs/my_training_run/test_results
```

## Outputs

```text
results_dir/
  coco_predictions.json
  paper_metrics.json
  test_summary.json
  test_summary.csv
  confidence_distribution.png
  confusion_matrix.png
```

The most useful files for reporting are:

```text
test_summary.json
test_summary.csv
coco_predictions.json
```

## Notes

- Repeating `test.py` on the same checkpoint and same test set is not a substitute for independent training runs.
- For run-to-run model variability, train multiple checkpoints with different seeds in `6-1_obj_det_trainer`, then evaluate each checkpoint once with this module.
- `test.py` imports shared model/dataset utilities from `../6-1_obj_det_trainer/train.py`, so keep the training and testing modules in the repository structure.

## License

This project is licensed under the MIT License. See the repository `LICENSE` file.

## Developer checks

```sh
python -m unittest discover -s tests
python tests/smoke_training_handoff.py
```

The smoke check trains a small CPU batch in Module 6.1, fine-tunes its checkpoint in Module 6.2, and runs the Module 7 command to produce metrics, predictions, and plots using temporary synthetic data. It verifies execution rather than model accuracy.
