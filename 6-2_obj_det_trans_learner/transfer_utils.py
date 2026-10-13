"""
Script Name: transfer_utils.py
Purpose: Provides utility functions for transfer learning with Faster R-CNN, including model loading, 
         layer freezing, optimizer configuration, and parameter management
Authors: Worasit Sangjan
Date: 15 February 2025 
Version: 1.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any

def read_checkpoint(checkpoint_path):
    """Read tensor checkpoints without enabling arbitrary pickle execution."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dictionary or contain model_state_dict.")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain model weights.")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint model weights must be tensors.")
    return checkpoint, state


def load_pretrained_model(model, checkpoint_path, checkpoint=None):
    """Load compatible model weights, failing rather than silently dropping them."""
    if checkpoint is None:
        checkpoint, state = read_checkpoint(checkpoint_path)
    else:
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
    expected = model.state_dict()
    wrong_shapes = [name for name, value in state.items() if name in expected and value.shape != expected[name].shape]
    if wrong_shapes:
        raise ValueError("Checkpoint architecture/classes do not match the YAML model: " + ", ".join(wrong_shapes[:5]))
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.endswith("num_batches_tracked")]
    if missing or result.unexpected_keys:
        raise ValueError("Checkpoint model keys do not match the selected architecture. Check model.architecture and anchor settings.")
    print("Pretrained model loaded successfully", flush=True)
    return checkpoint


def freeze_layers(model: nn.Module, config: Dict[str, Any]) -> None:
    """Freeze layers based on configuration."""
    def freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False
    
    if config['freeze_backbone']:
        print("Freezing backbone")
        freeze_params(model.backbone)
    
    if config['freeze_rpn']:
        print("Freezing RPN")
        freeze_params(model.rpn)
    
    if config['freeze_roi_heads']:
        print("Freezing ROI heads")
        freeze_params(model.roi_heads)

def get_transfer_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create an optimizer for transfer learning with different learning rates."""
    param_groups = []
    
    if not config['freeze_backbone']:
        param_groups.append({
            'params': model.backbone.parameters(),
            'lr': config['learning_rates']['backbone']
        })
    
    if not config['freeze_rpn']:
        param_groups.append({
            'params': model.rpn.parameters(),
            'lr': config['learning_rates']['rpn']
        })
    
    if not config['freeze_roi_heads']:
        param_groups.append({
            'params': model.roi_heads.parameters(),
            'lr': config['learning_rates']['roi_heads']
        })
    
    return torch.optim.SGD(
        param_groups,
        momentum=0.9,
        weight_decay=0.0005
    )

def print_trainable_parameters(model: nn.Module) -> None:
    """Print a summary of trainable parameters in the model."""
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%\n")