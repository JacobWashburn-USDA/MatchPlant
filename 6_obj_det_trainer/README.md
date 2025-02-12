# **Object Detection Training Tool**

This Python utility trains a Faster R-CNN model for object detection using PyTorch, with support for COCO format datasets. It provides configurable training parameters, data augmentation, and comprehensive evaluation metrics.

### **Note**
This tool is essential for training deep learning models on annotated datasets. It offers a robust implementation of Faster R-CNN with ResNet50 backbone, flexible configuration options, and detailed training metrics tracking, making it suitable for both research and practical applications.

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/Ortho_to_image.git
cd Ortho_to_image/6_obj_det_trainer
```

### 2. Download required utility files:
Download the following utility files from PyTorch's reference/detection repository and place them in your project directory:
```bash
# Using wget (or download manually from the URLs below)
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py
```

These files are essential for training and contain helper functions for:
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
# Edit config.yaml to match your dataset and requirements
python train.py
```

## **Features**

- Configurable Model Architecture: Customizable Faster R-CNN parameters
- Multi-Platform Support: CUDA, MPS (Apple Silicon), and CPU support
- Data Augmentation: Configurable image transformations
- Progress Tracking: Detailed training and validation metrics
- Checkpoint Management: Regular model saving and best model tracking
- Memory Management: Optimized for various hardware configurations
- COCO Evaluation: Standard object detection metrics

## **Requirements**

- Python 3.x
- Dependencies:
  - PyTorch
  - torchvision
  - pycocotools
  - PyYAML
  - Pillow
  - psutil

## **Input Requirements**

### **Required Files**
1. Training Dataset: This is the output from [5_img_splitterhere](https://github.com/JacobWashburn-USDA/Ortho_to_image/tree/main/5_img_splitter)
    - Images in standard formats (JPG, PNG, TIFF)
    - COCO format annotations (.json)
    - Organized in train/validation splits

2. Configuration File (config.yaml)

    Edit `config.yaml` to match your dataset and requirements:

    - Model Configuration:
       - Number of classes
       - Anchor sizes and ratios
       - Detection thresholds
       - RPN and ROI parameters

    - Training Configuration:
      - Learning rates
      - Batch size
      - Number of epochs
      - Checkpoint frequency
      - Data augmentation parameters

    - Resource Configuration:
      - CUDA settings
      - Memory management
      - Number of workers
  
    - Data: train and validation data location

Input structure:
```
project_root/
├── train.py              # Main training script
├── config.yaml           # Configuration file
├── config_loader.py      # Configuration loader
├── coco_eval.py          # COCO evaluation utilities
├── coco_utils.py         # COCO dataset utilities
├── engine.py             # Training engine
├── transforms.py         # Data transformation utilities
├── utils.py              # General utilities
├── data/
│   ├── train/            # Training images
│   │   ├── image1.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg    # Validation images
│       └── ...         
└── annotations/
    ├── train.json        # Training annotations (COCO format)
    └── val.json          # Validation annotations (COCO format)
```

## **Outputs**

1. Training and Validation Progress:

Real-time metric 

```
Epoch: [0][10/500]
boxes: 6
lr: 0.010000
loss: 1.2345
loss_classifier: 0.4567
loss_box_reg: 0.3456
```

Validation results
    
```
Evaluation Summary:
AP @ IoU=0.50:0.95: 0.456
AP @ IoU=0.50: 0.789
AP @ IoU=0.75: 0.567
```

2. Saving Results:
    - Checkpoints are saved in `./checkpoints/`
      - `best_model.pt`: Best performing model
      - `model_epoch_N.pt`: Regular checkpoints
      - `emergency_save.pt`: Save on training interruption

    - Validation results are saved in `./validation_results/`

Output structure:
```
project_root/
├── checkpoints/
│   ├── best_model.pt
│   ├── model_epoch_N.pt
│   └── emergency_save.pt
└── validation_results/
    ├── validation_results_epoch_N_timestamp.json
    └── validation_summary.jsonl
```

## **Usage Instructions**

1. Prepare Dataset:
   - Organize images in train/val folders
   - Ensure COCO format annotations
   - Verify file paths in config.yaml

2. Configure Training:
   - Adjust model parameters
   - Set training hyperparameters
   - Configure data augmentation
   - Specify resource settings

3. Start Training:
  ```bash
  python train.py
  ```

## Hardware Requirements

Minimum:
  - 8GB RAM
  - CUDA-capable GPU with 4GB VRAM (or Apple Silicon for MPS)
  - 20GB disk space

Recommended:
  - 16GB RAM
  - CUDA-capable GPU with 8GB+ VRAM
  - 50GB+ disk space

## Optimization Strategies

1. Memory Management:
    - Adjust batch size based on available GPU memory
    - Use `memory_config` in config.yaml
    - Enable gradient checkpointing for large models

2. Training Speed:
    - Use CUDA if available
    - Adjust the number of workers based on CPU cores
    - Enable mixed precision training for faster computation

3. Model Performance:
    - Customize anchor sizes to match object dimensions
    - Fine-tune IoU thresholds for improved detection accuracy
    - Apply appropriate data augmentations for better generalization


## Common Issues and Solutions

1. Out of Memory (OOM):
    - Reduce batch size
    - Decrease image dimensions
    - Enable gradient checkpointing

2. Slow Training:
    - Check GPU utilization
    - Increase num_workers if CPU bottlenecked
    - Enable mixed precision training

3. Poor Convergence:
    - Verify dataset annotations
    - Adjust learning rates
    - Check anchor sizes match your objects
    - Ensure proper class balance

## License

This project is licensed under the MIT License. For details, see the [LICENSE](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/LICENSE) file.
