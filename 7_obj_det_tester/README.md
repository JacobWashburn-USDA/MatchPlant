# **Object Detection Testing Tool**

This Python utility evaluates and tests Faster R-CNN models trained for object detection. It supports comprehensive evaluation metrics, statistical analysis across multiple test runs, and detailed result visualization.

### **Note**
This tool is designed for rigorous evaluation of trained deep learning models. It offers statistical analysis across multiple test runs, size-based performance evaluation, and detailed visualization capabilities, making it suitable for both research validation and practical deployment assessment.

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/Ortho_to_image.git
cd Ortho_to_image/7_obj_det_tester
```

### 2. Download required utility files (In case, these files are not downloaed in train process)

Download the following utility files from PyTorch's reference/detection repository and place them in your project directory:
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
    - Small object detection (0-32×32 pixels)
    - Medium object detection (32×32-96×96 pixels)
    - Large object detection (>96×96 pixels)
- Visualization Tools: Detection visualization, confidence distributions, confusion matrices
- Statistical Analysis: Mean, standard deviation, confidence intervals
- Memory Management: Optimized for various hardware configurations
- COCO Evaluation: Standard object detection metrics

## **Requirements**

- Python 3.x
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

### **Required Files**
1. Testing Dataset:
    - Test images in standard formats (`data/test/`)
    - COCO format annotations (`annotations/test.json`)
    - Trained model checkpoint (`checkpoints/best_model.pt`)

2. Configuration File (config.yaml)

    Edit `config.yaml` to match your requirements:

### Model Settings
```yaml
model_path: "./checkpoints/best_model.pt" # Model checkpoint path
num_runs: 5  # Number of test iterations
```

### Metrics Configuration
```yaml
metrics:
  iou_thresholds: [0.5, 0.75]
  confidence_threshold: 0.5
  size_ranges:
    small: [0, 1024]      # 32x32
    medium: [1024, 9216]  # 32x32 to 96x96
    large: [9216, null]   # > 96x96
```

### Visualization Settings
```yaml
visualization:
  confidence_threshold: 0.5
  pred_box_color: "red"
  gt_box_color: "blue"
  box_width: 2
  figure_size: [10, 6]
  dpi: 100
```

Input structure:
```
project_root/
├── test.py               # Main testing script
├── config.yaml           # Configuration file
├── config_loader.py      # Configuration loader
├── coco_eval.py          # COCO evaluation utilities
├── coco_utils.py         # COCO dataset utilities
├── engine.py             # Testing engine
├── transforms.py         # Data transformation utilities
├── utils.py              # General utilities
├── data/
│   └── test/            # Test images
│       ├── image1.jpg
│       └── ...         
├── annotations/
│   └── test.json        # Test annotations (COCO format)
└── checkpoints/
    └── best_model.pt    # Trained model
```

## Output Structure

1. Performance Metrics:

```
IoU_0.5:
Precision: 0.856 ± 0.023 (std: 0.012)
Recall: 0.789 ± 0.018 (std: 0.009)
F1_Score: 0.821 ± 0.015 (std: 0.008)
```

2. Size-based Analysis:
```
Small Objects:
Precision: 0.723 ± 0.034
Recall: 0.654 ± 0.028
F1_Score: 0.687 ± 0.025
```

Output structure:
```
project_root/
├── paper_results/
│   ├── run_1/
│   │   ├── metrics_run_1.json
│   │   ├── confidence_distribution.png
│   │   └── confusion_matrix.png
│   ├── run_2/
│   │   └── ...
│   └── aggregate_results.json
└── test_results/
    ├── run_1/
    │   └── test_image_*.png
    └── run_2/
        └── ...
```

## **Usage Instructions**

1. Prepare Test Environment:
   - Place trained model in checkpoints directory
   - Organize test images and annotations
   - Verify file paths in test_config.yaml

2. Configure Testing:
   - Set model checkpoint path
   - Configure number of test runs
   - Set evaluation metrics
   - Configure visualization options

3. Run Tests:
  ```bash
  python test.py
  ```

## Hardware Requirements

Minimum:
- 8GB RAM
- CUDA-capable GPU with 4GB VRAM
- 20GB disk space

Recommended:
- 16GB RAM
- CUDA-capable GPU with 8GB+ VRAM
- 50GB+ disk space

## Optimization Strategies

1. Memory Usage:
    - Adjust batch size based on available GPU memory
    - Optimize number of workers
    - Use appropriate DPI for visualizations

2. Testing Speed:
    - Use CUDA if available
    - Adjust worker count based on CPU cores
    - Consider reducing visualization DPI for faster processing

3. Result Quality:
    - Adjust confidence thresholds for optimal precision/recall balance
    - Configure size ranges based on your specific use case
    - Increase number of test runs for more reliable statistics

## Common Issues and Solutions

1. Memory Issues
   - Reduce batch size in config.yaml
   - Decrease number of workers
   - Run on smaller subset of test data

2. Performance Issues
   - Check GPU utilization
   - Optimize number of workers
   - Adjust confidence thresholds

3. Visualization Problems
   - Ensure output directories exist
   - Check file permissions
   - Verify image format compatibility


## License

This project is licensed under the MIT License. For details, see the [LICENSE](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/LICENSE) file.