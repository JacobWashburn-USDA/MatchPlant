# Transfer Learning Guide for Object Detection

This guide explains the transfer learning implementation in our object detection system, focusing on layer freezing, code reuse, and configuration options.


## Dashboard configuration and implementation

The [Transfer Learning README](README.md#using-the-dashboard) explains the settings form, Browse buttons, help icons, and advanced YAML editor. The form writes `transfer_config.yaml` before running the Python training script.

Module 6.2 directly loads shared helpers from the sibling Module 6.1 folder. Keep both folders in the repository; do not copy training scripts into Module 6.2. Its own configuration loader and model builder bridge the current APIs.

## Table of Contents
- [Model Architecture and Freezing](#model-architecture-and-freezing)
- [Code Reuse from Base Training](#code-reuse-from-base-training)
- [When to Freeze/Unfreeze Layers](#when-to-freezeunfreeze-layers)
- [Configuration Examples](#configuration-examples)

## Model Architecture and Freezing

Our Faster R-CNN model consists of three main components that can be selectively frozen:

- ### Backbone Network (ResNet50)
  ```python
  model.backbone
  ```
  - **Purpose**: Extracts general features from images
  - **Layers**: Multiple convolutional layers organized in blocks
  - **What it learns**: 
    - Basic features (edges, textures)
    - Complex patterns
    - Object parts
  - **Freezing code**:
    ```python
    # In transfer_utils.py
    if config['freeze_backbone']:
        for param in model.backbone.parameters():
            param.requires_grad = False
    ```
- ### Region Proposal Network (RPN)
  ```python
  model.rpn
  ```
  - **Purpose**: Generates region proposals where objects might be
  - **Components**:
    - Intermediate layer
    - Classification layer (objectness)
    - Regression layer (box coordinates)
  - **Freezing code**:
    ```python
    if config['freeze_rpn']:
        for param in model.rpn.parameters():
            param.requires_grad = False
    ```
- ### ROI Heads
  ```python
  model.roi_heads
  ```
  - **Purpose**: Refines proposals and classifies objects
  - **Components**:
    - Box predictor
    - Classification head
    - Regression head
  - **Freezing code**:
    ```python
    if config['freeze_roi_heads']:
        for param in model.roi_heads.parameters():
            param.requires_grad = False
    ```

## Code Reuse from Base Training

`transfer_train.base_training()` loads `../6-1_obj_det_trainer/train.py` explicitly. The transfer runner reuses `MaizeDatasetCOCO`, `get_transform(train)`, `collate_fn`, `configure_device_and_resources()`, and `evaluate(...)`.

Module 6.2 provides its own validated YAML loader, checkpoint-aware Faster R-CNN builder, and training loop. It does not depend on `train_config_loader.py` or Module 6.1's `build_model` signature. Model architecture and anchor count are checked against checkpoint weights; custom anchor dimensions must match the checkpoint's original configuration. See [checkpoint compatibility](README.md#checkpoint-compatibility).

The frozen backbone stays in evaluation mode during training so its normalization statistics are preserved. RPN and ROI heads stay in the training loss path even when their parameters are frozen. At least one component must be unfrozen.

## When to Freeze/Unfreeze Layers

Here are common scenarios and recommended freezing strategies:

- ### Similar Domain Transfer
  When transferring between similar objects (e.g., maize to wheat):
  ```yaml
  transfer:
    freeze_backbone: true     # Keep learned features
    freeze_rpn: false         # Adapt to new object shapes
    freeze_roi_heads: false   # Learn new class characteristics
  ```
- ### Different Domain Transfer
  When transferring to very different objects:
  ```yaml
  transfer:
    freeze_backbone: false    # Allow feature adaptation
    freeze_rpn: false         # Learn new object proposals
    freeze_roi_heads: false   # Learn new classifications
  ```
- ### Small Dataset Transfer
  When fine-tuning with limited data:
  ```yaml
  transfer:
    freeze_backbone: true     # Prevent overfitting
    freeze_rpn: true          # Minimize parameters
    freeze_roi_heads: false   # Only adapt classification
  ```
- ### Large Dataset Transfer
  When fine-tuning with abundant data:
  ```yaml
  transfer:
    freeze_backbone: false    # Learn domain-specific features
    freeze_rpn: false         # Full adaptation
    freeze_roi_heads: false   # Learn new objects fully
  ```

## Configuration Examples

Example configurations for different scenarios:

- ### Fine-tuning for Similar Plants (Default Size)
  ```yaml
  transfer:
    freeze_backbone: true
    freeze_rpn: false
    freeze_roi_heads: false
    learning_rates:
      backbone: 0.0001
      rpn: 0.001
      roi_heads: 0.001
    training:
      epochs: 30
      batch_size: 1
  model:
    min_size: 800
    max_size: 1333
    anchor_sizes: [[4, 8, 16, 32, 64, 96, 128, 256]]
  ```

- ### Using Pre-trained Model with Different Image Size (e.g., 640x640)
  ```yaml
  transfer:
    freeze_backbone: true
    freeze_rpn: false
    freeze_roi_heads: false
    learning_rates:
      backbone: 0.0001
      rpn: 0.001
      roi_heads: 0.001
    training:
      epochs: 30
      batch_size: 1
  model:
    min_size: 640
    max_size: 640
    # Adjusted anchor sizes for 640x640 images
    anchor_sizes: [[4, 8, 16, 32, 64, 96, 128]]
    anchor_ratios: [[0.5, 1.0, 2.0]]
  ```

- ### Limited Data Scenario
  ```yaml
  transfer:
    freeze_backbone: true
    freeze_rpn: true
    freeze_roi_heads: false
    learning_rates:
      backbone: 0.0
      rpn: 0.0
      roi_heads: 0.001
    training:
      epochs: 20
      batch_size: 1
  ```

## Tips for Transfer Learning

1. **Monitor Training**:
   - Watch validation mAP
   - Check for overfitting
   - Monitor gradient magnitudes
2. **Learning Rate Selection**:
   - Use lower rates for unfrozen backbone
   - Higher rates for ROI heads
   - Adjust based on validation performance
3. **Data Considerations**:
   - Image Size Compatibility:
     * New training images should typically match the size specifications used in your pre-trained model
     * If using our default model: min_size=800, max_size=1333
     * If using a different pre-trained model:
       - Check the image sizes used in the pre-trained model's training
       - Adjust `model.min_size` and `model.max_size` in `transfer_config.yaml` to match
       - Example for a model trained on 640x640 images:
         ```yaml
         model:
           min_size: 640
           max_size: 640
         ```
     * Key considerations when changing image sizes:
       - Anchor sizes may need adjustment to match new image dimensions
       - Memory usage will change with image size
       - Training time will be affected
       - Detection performance might vary, especially for small objects
   - Dataset Size:
     * More data = More layers can be unfrozen
     * Similar domain = More layers can be frozen
     * Different domain = Consider unfreezing more layers
   - Image Characteristics:
     * Similar image characteristics (brightness, contrast, etc.) to pre-trained data improve transfer
     * Consider normalizing your images to match pre-trained data statistics
   
   Note: When using a pre-trained model with different image sizes, you might need to adjust the anchor sizes (`model.anchor_sizes`) to match better the new image dimensions and your target object sizes.
4. **Evaluation**:
   - Compare with a baseline model
   - Check per-class performance
   - Validate diverse test cases
