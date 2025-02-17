# **Object Detection Transfer Learning Tool**

This Python utility uses PyTorch to implement transfer learning for Faster R-CNN models in object detection. It enables fine-tuning pre-trained models with configurable layer freezing, learning rates, and comprehensive evaluation metrics. The pre-trained model is provided, trained on individual maize in a research field.

### **Note**

This tool extends the base object detection training system with transfer learning capabilities. It provides flexible configuration for layer freezing, learning rate adjustments, and selective parameter updates, making it ideal for adapting pre-trained models to new domains with limited data.

See [Transfer Learning Guide](transfer_learning_guide.md) for detailed technical information about the transfer learning implementation.

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
cd Ortho_to_image/6-2_obj_det_trans_learner
```

### 2. Download required files:
- Pre-trained model: Please download from [Zenodo]? 
- Required base files: from [6-1_obj_det_trainer](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/6-1_obj_det_trainer)
    - Base training script: `train.py`
    - Configuration loader: `train_config_loader.py`
    - PyTorch utility files: `coco_eval.py`, `coco_utils.py`, `engine.py`, `transforms.py`, and `utils.py`.
  
### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Configure and run:
Edit `transfer_config.yaml` to set your transfer learning parameters. For detailed configuration options, see the [Transfer Learning Guide - Configuration Examples](transfer_learning_guide.md#configuration-examples):
```bash
python transfer_train.py
```

## **Features**

- Selective Layer Freezing: Control which model components to freeze/unfreeze, see [Transfer Learning Guide - Model Architecture and Freezing](transfer_learning_guide.md#model-architecture-and-freezing)
- Component-Specific Learning Rates: Different learning rates for backbone, RPN, and ROI heads
- Pre-trained Model Loading: Robust checkpoint loading with error handling
- Parameter Management: Detailed tracking of trainable parameters
- Configurable Transfer Settings: Flexible configuration through YAML
- Progress Tracking: Comprehensive training and validation metrics
- Checkpoint Management: Regular and best model saving
- COCO Evaluation: Standard object detection metrics

## **Requirements**

- Python 3.9+
- Dependencies:
  - PyTorch
  - torchvision
  - pycocotools
  - PyYAML
  - Pillow
  - psutil

## **Input Requirements**

### 1. Dataset Requirements
- **Image Size Compatibility**: If the provided pre-trained model is used, your new training images should have similar dimensions to those used in the pre-trained model. For more detail, please check [Transfer Learning Guide - Tips for Transfer Learning](transfer_learning_guide.md#tips-for-transfer-learning) and [Transfer Learning Guide - Configuration Examples](transfer_learning_guide.md#configuration-examples)
  - The model configuration in `transfer_config.yaml` specifies (In case the provided pre-trained model is used):
    ```yaml
    model:
      min_size: 800
      max_size: 1333
    ```
  - Your new training images should be prepared to match these dimensions. Significant differences in image sizes between pre-trained and new data can negatively impact transfer learning performance
- **Dataset**:
  - New domain images and annotations in COCO format: Use [4_bbox_drawer](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/4_bbox_drawer)
  - Organized in train/validation splits: Use [5_img_splitter](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/5_img_splitter)

### 2. Required Files
- Pre-trained Model: Download from [Zenodo](?)
- Base training and utility codes: Download from [6-1_obj_det_trainer](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/6-1_obj_det_trainer)
  - Base training codes: `train.py` and `train_config_loader.py`
  - Utility files: `coco_eval.py`, `coco_utils.py`, `engine.py`, `transforms.py`, and `utils.py`
- Transfer Learning Configuration:
  - `transfer_config.yaml` (see [Configuration Examples](transfer_learning_guide.md#configuration-examples))
- Code Files:
  - `transfer_train.py`: Main transfer learning script
  - `transfer_utils.py`: Transfer learning utilities
- Input structure:
  ```
  project_root/
  ├── transfer_train.py           # Transfer learning main script
  ├── transfer_utils.py           # Transfer learning utilities
  ├── transfer_config.yaml        # Transfer learning configuration
  ├── pre-trained model.pt        # Pre-trained model checkpoint
  ├── train.py          
  ├── train_config_loader.py      
  ├── coco_eval.py        
  ├── coco_utils.py
  ├── engine.py     
  ├── transforms.py       
  ├── utils.py        
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
  - Real-time metric  
    ```
    Epoch: [0][10/500]
    boxes: 6
    lr: 0.010000
    loss: 1.2345
    loss_classifier: 0.4567
    loss_box_reg: 0.3456
    loss_objectness: 0.2345
    loss_rpn_box_reg: 0.5678
    ```
  - Validation results 
    ```
    Evaluation Summary:
    AP @ IoU=0.50:0.95: 0.456
    AP @ IoU=0.50: 0.789
    AP @ IoU=0.75: 0.567
    AP for small/medium/large objects 
    AR (Average Recall) metrics 
    ```
- Saving Results:
  - Checkpoints are saved in `./checkpoints/`
    - `best_model_transfer.pt`: Best performing model
    - `model_transfer_epoch_N.pt`: Regular checkpoints
    - `emergency_save_transfer.pt`: Save on interruption
    - Validation results are saved in `./validation_results/`
- Output structure:
  ```
  project_root/
  ├── checkpoints/
  │   ├── best_model_transfer.pt
  │   ├── model_transfer_epoch_N.pt
  │   └── emergency_save_transfer.pt
  └── validation_results/
      ├── validation_results_epoch_N_timestamp.json
      └── validation_summary.jsonl
  ```
  
## **Usage Instructions**

- Prepare Pre-trained Model:
   - Ensure the base model is appropriately trained
- Configure Transfer Learning:
   - Set layer freezing strategy (see [When to Freeze/Unfreeze Layers](transfer_learning_guide.md#when-to-freezeunfreeze-layers))
   - Configure learning rates
   - Specify dataset paths
- Start Transfer Learning:
  ```bash
  python transfer_train.py
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

For detailed optimization guidelines, see [Tips for Transfer Learning](transfer_learning_guide.md#tips-for-transfer-learning).

- Layer Freezing:
  - Freeze backbone for similar domains
  - Unfreeze more layers for different domains
  - Consider dataset size when choosing a freezing strategy
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
