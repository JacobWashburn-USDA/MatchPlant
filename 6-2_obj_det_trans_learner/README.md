# **Object Detection Transfer Learning Tool**

This Python utility implements transfer learning for Faster R-CNN models in object detection using PyTorch. It enables fine-tuning of pre-trained models with configurable layer freezing, learning rates, and comprehensive evaluation metrics.

### **Note**

This tool extends the base object detection training system with transfer learning capabilities. It provides flexible configuration for layer freezing, learning rate adjustments, and selective parameter updates, making it ideal for adapting pre-trained models to new domains with limited data.

For detailed technical information about the transfer learning implementation, see [Transfer Learning Guide](transfer_learning_guide.md).

## Table of Contents
- [**Object Detection Transfer Learning Tool**](#object-detection-transfer-learning-tool)
    - [**Note**](#note)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [1. Prepare your environment:](#1-prepare-your-environment)
    - [2. Configure transfer learning:](#2-configure-transfer-learning)
    - [3. Run transfer learning:](#3-run-transfer-learning)
  - [**Features**](#features)
  - [**Requirements**](#requirements)
  - [**Input Requirements**](#input-requirements)
    - [1. Required Files](#1-required-files)
  - [**Outputs**](#outputs)
  - [**Usage Instructions**](#usage-instructions)
  - [Hardware Requirements](#hardware-requirements)
  - [Optimization Strategies](#optimization-strategies)
  - [Common Issues and Solutions](#common-issues-and-solutions)
  - [License](#license)

## Quick Start

### 1. Prepare your environment:
Make sure you have the base object detection training system installed and configured as described in the [base README](README.md).

### 2. Configure transfer learning:
Edit `transfer_config.yaml` to set your transfer learning parameters. For detailed configuration options, see the [Transfer Learning Guide - Configuration Examples](transfer_learning_guide.md#configuration-examples):

```yaml
transfer:
  pretrained_model_path: "best_model.pt"
  freeze_backbone: true
  freeze_rpn: false
  freeze_roi_heads: false
  learning_rates:
    backbone: 0.0001
    rpn: 0.001
    roi_heads: 0.001
```

### 3. Run transfer learning:
```bash
python transfer_train.py
```

## **Features**

- Selective Layer Freezing: Control which model components to freeze/unfreeze (see [Model Architecture and Freezing](transfer_learning_guide.md#model-architecture-and-freezing))
- Component-Specific Learning Rates: Different learning rates for backbone, RPN, and ROI heads
- Pre-trained Model Loading: Robust checkpoint loading with error handling
- Parameter Management: Detailed tracking of trainable parameters
- Configurable Transfer Settings: Flexible configuration through YAML
- Progress Tracking: Comprehensive training and validation metrics
- Checkpoint Management: Regular and best model saving
- COCO Evaluation: Standard object detection metrics

## **Requirements**

Same as base training system (see [base requirements](README.md#requirements)), plus:
- Python 3.9+
- PyTorch
- torchvision
- pycocotools
- PyYAML
- Pillow
- psutil

## **Input Requirements**

### 1. Required Files
- Pre-trained Model:
  - Checkpoint file (.pt) from base training
- Transfer Learning Configuration:
  - `transfer_config.yaml` (see [Configuration Examples](transfer_learning_guide.md#configuration-examples))
- Dataset:
  - New domain images and annotations in COCO format
  - Organized in train/validation splits
- Code Files:
  - `transfer_train.py`: Main transfer learning script
  - `transfer_utils.py`: Transfer learning utilities
  - Base training system utilities (see [Code Reuse from Base Training](transfer_learning_guide.md#code-reuse-from-base-training))
- Input structure:
  ```
  project_root/
  ├── transfer_train.py           # Transfer learning main script
  ├── transfer_utils.py           # Transfer learning utilities
  ├── transfer_config.yaml        # Transfer learning configuration
  ├── best_model.pt               # Pre-trained model checkpoint
  ├── data/
  │   ├── new_train/              # New domain training images
  │   │   ├── image1.jpg
  │   │   └── ...
  │   └── new_val/                # New domain validation images
  │       ├── image1.jpg
  │       └── ...         
  └── annotations/
      ├── new_train.json          # New domain training annotations
      └── new_val.json            # New domain validation annotations
  ```

## **Outputs**

- Training and Validation Progress:
  - Real-time metrics including:
    ```
    Epoch: [0][10/500]
    Trainable parameters: 1,234,567
    Total parameters: 2,345,678
    Percentage trainable: 52.63%
    Loss: 1.2345
    Validation mAP: 0.789
    ```
- Saved Models:
  - Checkpoints saved in `./checkpoints/`
    - `best_model_transfer.pt`: Best performing model
    - `model_transfer_epoch_N.pt`: Regular checkpoints
    - `emergency_save_transfer.pt`: Save on interruption

## **Usage Instructions**

- Prepare Pre-trained Model:
   - Ensure base model is properly trained
   - Note the checkpoint path
- Configure Transfer Learning:
   - Set layer freezing strategy (see [When to Freeze/Unfreeze Layers](transfer_learning_guide.md#when-to-freezeunfreeze-layers))
   - Configure learning rates
   - Specify dataset paths
- Start Transfer Learning:
  ```bash
  python transfer_train.py
  ```

## Hardware Requirements

Same as base training system (see [base hardware requirements](README.md#hardware-requirements)):
- Minimum:
  - 8GB RAM
  - CUDA-capable GPU with 4GB VRAM
  - 20GB disk space
- Recommended:
  - 16GB RAM
  - CUDA-capable GPU with 8GB+ VRAM
  - 50GB+ disk space

## Optimization Strategies

For detailed optimization guidelines, see [Tips for Transfer Learning](transfer_learning_guide.md#tips-for-transfer-learning).

- Layer Freezing:
  - Freeze backbone for similar domains
  - Unfreeze more layers for different domains
  - Consider dataset size when choosing freezing strategy
- Learning Rates:
  - Use lower rates for unfrozen backbone
  - Higher rates for ROI heads
  - Adjust based on validation performance
- Data Requirements:
  - More data = More layers can be unfrozen
  - Similar domain = More layers can be frozen
  - Different domain = Consider unfreezing more layers

## Common Issues and Solutions

- Poor Transfer Performance:
  - Verify pre-trained model quality
  - Adjust layer freezing strategy (see [When to Freeze/Unfreeze Layers](transfer_learning_guide.md#when-to-freezeunfreeze-layers))
  - Fine-tune learning rates
  - Ensure sufficient training data
- Memory Issues:
  - Reduce batch size
  - Freeze more layers
  - Enable gradient checkpointing
- Slow Convergence:
  - Check learning rates
  - Verify domain similarity
  - Adjust freezing strategy
  - Monitor gradient flow

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.