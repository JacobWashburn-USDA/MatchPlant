"""
Script Name: train_config_loader
Purpose: To manage configuration for object detection training.
Author: Worasit Sangjan
Date: 11 Febuary 2025
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

@dataclass
class AugmentationConfig:
    """Data augmentation settings."""
    horizontal_flip_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2

    def __post_init__(self):
        if not 0 <= self.horizontal_flip_prob <= 1:
            raise ValueError("horizontal_flip_prob must be between 0 and 1")
        if not 0 <= self.brightness <= 1:
            raise ValueError("brightness must be between 0 and 1")
        if not 0 <= self.contrast <= 1:
            raise ValueError("contrast must be between 0 and 1")
        if not 0 <= self.saturation <= 1:
            raise ValueError("saturation must be between 0 and 1")

@dataclass
class OptimizerConfig:
    """Optimizer settings."""
    backbone_lr: float = 0.005
    rpn_lr: float = 0.01
    roi_heads_lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 10
    lr_gamma: float = 0.1

    def __post_init__(self):
        if any(lr <= 0 for lr in [self.backbone_lr, self.rpn_lr, self.roi_heads_lr]):
            raise ValueError("Learning rates must be positive")
        if not 0 <= self.momentum <= 1:
            raise ValueError("Momentum must be between 0 and 1")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")

@dataclass
class ModelConfig:
    """Model architecture and detection parameters."""
    num_classes: int
    min_size: int = 800
    max_size: int = 1333
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.4
    box_detections_per_img: int = 100
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 1000
    rpn_post_nms_top_n_test: int = 500
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    box_fg_iou_thresh: float = 0.6
    box_bg_iou_thresh: float = 0.3
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.35
    anchor_sizes: Tuple[Tuple[int, ...], ...] = ((4, 8, 16, 32, 64, 96, 128, 256),)
    anchor_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 0.75, 1.0, 1.5, 2.0),)

    def __post_init__(self):
        if self.num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if not 0 <= self.box_score_thresh <= 1:
            raise ValueError("box_score_thresh must be between 0 and 1")
        if not 0 <= self.box_nms_thresh <= 1:
            raise ValueError("box_nms_thresh must be between 0 and 1")
        if not all(isinstance(size, tuple) for size in self.anchor_sizes):
            raise ValueError("anchor_sizes must be a tuple of tuples")
        if not all(isinstance(ratio, tuple) for ratio in self.anchor_ratios):
            raise ValueError("anchor_ratios must be a tuple of tuples")

@dataclass
class TrainingConfig:
    """Training process settings."""
    epochs: int = 150
    batch_size: int = 1
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.save_freq < 1:
            raise ValueError("save_freq must be >= 1")

@dataclass
class DataConfig:
    """Dataset and data loading settings."""
    train_dir: str
    val_dir: str
    train_annotations: str
    val_annotations: str
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

@dataclass
class ResourceConfig:
    """Hardware resource settings."""
    memory_config: str = "max_split_size_mb:32"
    cuda_benchmark: bool = True

class ConfigValidator:
    """Validates configuration settings."""
    
    @staticmethod
    def validate_paths(config: DataConfig) -> None:
        """Validate dataset paths."""
        paths = [
            Path(config.train_dir),
            Path(config.val_dir),
            Path(config.train_annotations),
            Path(config.val_annotations)
        ]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

class ConfigLoader:
    """Configuration loader."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    @staticmethod
    def create_configs(config_dict: Dict[str, Any]) -> Tuple[ModelConfig, TrainingConfig, DataConfig, ResourceConfig]:
        """Create and validate configuration objects."""
        try:
            model_config = ModelConfig(**config_dict['model'])
            training_config = TrainingConfig(**config_dict['training'])
            data_config = DataConfig(**config_dict['data'])
            resource_config = ResourceConfig(**config_dict.get('resources', {}))
            
            ConfigValidator.validate_paths(data_config)
            
            return model_config, training_config, data_config, resource_config
        except KeyError as e:
            raise ValueError(f"Missing required configuration section: {e}")
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameter: {e}")

    @staticmethod
    def save_config(config_dict: Dict[str, Any], output_path: str) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)