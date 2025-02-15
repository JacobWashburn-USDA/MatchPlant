# **Object Detection Testing Tool**

This Python utility evaluates and tests Faster R-CNN models trained for object detection. It supports comprehensive evaluation metrics, statistical analysis across multiple test runs, and detailed result visualization.

### **Note**

This tool is designed to evaluate trained deep-learning models rigorously. It offers statistical analysis across multiple test runs, size-based performance evaluation, and detailed visualization capabilities, making it suitable for research validation and practical deployment assessment.

## Table of Contents
- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Input Requirements](#input-requirements)
- [Outputs](#outputs)
- [Usage Instructions](#usage-instructions)
- [Hardware Requirements](#hardware-requirements)
- [Optimization Strategies](#optimization-strategies)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [License](#license)

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/MatchPlant.git
cd Ortho_to_image/7_obj_det_tester
```

### 2. Download required utility files (In case these files are not downloaded in train process)

These utility files are essential for the testing tool to function. Without these files, the testing process will not work:
```bash
# Using wget (or download manually from the URLs below)
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py
```
These files are essential for training and testing, containing helper functions for:
- COCO dataset evaluation (coco_eval.py, coco_utils.py)
- Training engine and utilities (engine.py)
- Data transformations and augmentation (transforms.py)
- General utilities (utils.py)

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configure and run:

```bash
# Edit test_config.yaml to match your test requirements
python test.py
```

## **Features**

- Multiple Test Iterations: Statistical analysis across multiple runs
- Comprehensive Metrics: IoU-based evaluation at multiple thresholds
- Size-based Analysis: Performance evaluation for small, medium, and large objects
    - COCO Evaluation: Standard object detection metrics
        - Small object detection (0-32×32 pixels)
        - Medium object detection (32×32-96×96 pixels)
        - Large object detection (>96×96 pixels)
- Visualization Tools: Detection visualization, confidence distributions, confusion matrices
- Statistical Analysis: Mean, standard deviation, confidence intervals

## **Requirements**

- Python 3.9+
- Dependencies:
  - PyTorch
  - torchvision
  - pycocotools
  - PyYAML
  - matplotlib
  - seaborn
  - Pillow
  - numpy

## **Input Requirements**

### 1. Required Files
- Testing Dataset:
    - Test images in standard formats (`data/test/`) from [5_img_splitter](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/5_img_splitter)
    - COCO format annotations (`annotations/test.json`) from [5_img_splitter](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/5_img_splitter)
    - Trained model checkpoint (`checkpoints/best_model.pt`) from [6_obj_det_trainer](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/6_obj_det_trainer)
- PyTorch Utility Files:
    - coco_eval.py, coco_utils.py, engine.py, transforms.py, and utils.py
- Configuration File: Edit `test_config.yaml` to match your requirements:
    - Model Settings
      ```yaml
      model_path: "./checkpoints/best_model.pt" # Model checkpoint path
      num_runs: 5  # Number of test iterations
      ```
    - Metrics Configuration
      ```yaml
      metrics:
          iou_thresholds: [0.5, 0.75]
          confidence_threshold: 0.5
          size_ranges:
            small: [0, 1024]      # 32x32
            medium: [1024, 9216]  # 32x32 to 96x96
            large: [9216, null]   # > 96x96
      ```
    - Visualization Settings
      ```yaml
      visualization:
          confidence_threshold: 0.5
          pred_box_color: "red"
          gt_box_color: "blue"
          box_width: 2
          figure_size: [10, 6]
          dpi: 100
      ```
- Input structure:
  ```
  project_root/
    ├── test.py                     # Main testing script
    ├── test_config.yaml            # Configuration file
    ├── test_config_loader.py       # Configuration loader
    ├── coco_eval.py                # COCO evaluation utilities
    ├── coco_utils.py               # COCO dataset utilities
    ├── engine.py                   # Testing engine
    ├── transforms.py               # Data transformation utilities
    ├── utils.py                    # General utilities
    ├── data/
    │   └── test/                   # Test images
    │       ├── image1.jpg
    │       └── ...         
    ├── annotations/
    │   └── test.json               # Test annotations (COCO format)
    └── checkpoints/
        └── best_model.pt           # Trained model
  ```

## Outputs

The tool generates comprehensive evaluation results, including:
- Mean Average Precision (mAP) Metrics:
  ```
  AP_IoU=0.50:0.95: 0.856 ± 0.023 (std: 0.012, min: 0.834, max: 0.878)
  AP_IoU=0.50: 0.923 ± 0.015 (std: 0.008, min: 0.915, max: 0.931)
  AP_IoU=0.75: 0.867 ± 0.019 (std: 0.010, min: 0.857, max: 0.877)
  ```
- Size-based AP Metrics:
  ```
  AP_small: 0.723 ± 0.034 (std: 0.017, min: 0.706, max: 0.740)
  AP_medium: 0.845 ± 0.025 (std: 0.013, min: 0.832, max: 0.858)
  AP_large: 0.891 ± 0.021 (std: 0.011, min: 0.880, max: 0.902)
  ```
- Average Recall (AR) Metrics:
  ```
  AR_maxDets=1: 0.654 ± 0.028 (std: 0.014, min: 0.640, max: 0.668)
  AR_maxDets=10: 0.789 ± 0.023 (std: 0.012, min: 0.777, max: 0.801)
  AR_maxDets=100: 0.821 ± 0.019 (std: 0.010, min: 0.811, max: 0.831)
  ```
- IoU-based Performance:
  ```
  IoU_0.5:
      Precision: 0.856 ± 0.023 (std: 0.012, min: 0.844, max: 0.868)
      Recall: 0.789 ± 0.018 (std: 0.009, min: 0.780, max: 0.798)
      F1_Score: 0.821 ± 0.015 (std: 0.008, min: 0.813, max: 0.829)

  IoU_0.75:
      Precision: 0.823 ± 0.025 (std: 0.013, min: 0.810, max: 0.836)
      Recall: 0.754 ± 0.021 (std: 0.011, min: 0.743, max: 0.765)
      F1_Score: 0.787 ± 0.018 (std: 0.009, min: 0.778, max: 0.796)
  ```
- Size-based Performance Analysis:
  ```
  Small Objects:
      Precision: 0.723 ± 0.034 (std: 0.017, min: 0.706, max: 0.740)
      Recall: 0.654 ± 0.028 (std: 0.014, min: 0.640, max: 0.668)
      F1_Score: 0.687 ± 0.025 (std: 0.013, min: 0.674, max: 0.700)

  Medium Objects:
      Precision: 0.845 ± 0.025 (std: 0.013, min: 0.832, max: 0.858)
      Recall: 0.789 ± 0.023 (std: 0.012, min: 0.777, max: 0.801)
      F1_Score: 0.816 ± 0.019 (std: 0.010, min: 0.806, max: 0.826)

  Large Objects:
      Precision: 0.891 ± 0.021 (std: 0.011, min: 0.880, max: 0.902)
      Recall: 0.854 ± 0.019 (std: 0.010, min: 0.844, max: 0.864)
      F1_Score: 0.872 ± 0.015 (std: 0.008, min: 0.864, max: 0.880)
  ```
- Output structure:
  ```
  project_root/
    ├── paper_results/                          # Primary results directory
    │   ├── run_1/
    │   │   ├── metrics_run_1.json              # Detailed metrics for run 1
    │   │   │   ├── map_metrics                 # mAP and AR metrics
    │   │   │   ├── performance                 # IoU-based metrics
    │   │   │   └── size_performance            # Size-based metrics
    │   │   ├── confidence_distribution.png     # Distribution of detection scores
    │   │   └── confusion_matrix.png            # TP, FP, FN analysis
    │   ├── run_2/
    │   │   └── ...
    │   └── aggregate_results.json              # Statistical analysis across all runs
    │       ├── mean                            # Mean values for all metrics
    │       ├── std                             # Standard deviations
    │       ├── min                             # Minimum values
    │       ├── max                             # Maximum values
    │       └── 95_confidence                   # Confidence intervals
    └── test_results/                           # Detection visualization results
        ├── run_1/
        │   └── test_image_*.png                # Individual detection results
        └── run_2/
            └── ...
  ```

## **Usage Instructions**

- Prepare Test Environment:
   - Check the trained model in the checkpoints directory
   - Check test images and annotations
   - Verify PyTorch utility files are downloaded
   - Verify file paths in test_config.yaml
- Configure Testing:
   - Set model checkpoint path
   - Configure the number of test runs
   - Set evaluation metrics
   - Configure visualization options
- Run Tests:
  ```bash
  python test.py
  ```

## Hardware Requirements

- Minimum:
   - 8GB RAM
   - CUDA-capable GPU with 4GB VRAM
   - 20GB disk space
- Recommended:
   - 16GB RAM
   - CUDA-capable GPU with 8GB+ VRAM
   - 50GB+ disk space

## Optimization Strategies

- Memory Usage:
   - Adjust batch size based on available GPU memory
   - Optimize the number of workers
   - Use appropriate DPI for visualizations
- Testing Speed:
   - Use CUDA if available
   - Adjust worker count based on CPU cores
   - Consider reducing visualization DPI for faster processing
- Result Quality:
   - Adjust confidence thresholds for optimal precision/recall balance
   - Configure size ranges based on your specific use case
   - Increase the number of test runs for more reliable statistics

## Common Issues and Solutions

- Memory Issues
   - Reduce batch size in test_config.yaml
   - Decrease the number of workers
   - Run on a smaller subset of test data
- Performance Issues
   - Check GPU utilization
   - Optimize the number of workers
   - Adjust confidence thresholds
- Visualization Problems
   - Ensure output directories exist
   - Check file permissions
   - Verify image format compatibility
- PyTorch Utility Issues
   - Verify all required utility files are downloaded
   - Check file permissions of utility files
   - Ensure utility files are in the correct directory

## License

This project is licensed under the MIT License. For details, see the [LICENSE](https://github.com/JacobWashburn-USDA/MathPlant/blob/main/LICENSE) file.
