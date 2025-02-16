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

def load_pretrained_model(model: nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    """Load a pretrained model checkpoint."""
    print(f"Loading pretrained model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print("Pretrained model loaded successfully")
    
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