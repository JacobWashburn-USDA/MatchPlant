import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    num_classes: int
    anchor_sizes: Tuple[Tuple[int, ...], ...]
    anchor_ratios: Tuple[Tuple[float, ...], ...]
    min_size: int = 980
    max_size: int = 1240
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100
    backbone: str = "resnet50_fpn_v2"

@dataclass
class TrainingConfig:
    epochs: int = 150
    batch_size: int = 1
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 5

@dataclass
class DataConfig:
    train_dir: str
    val_dir: str
    train_annotations: str
    val_annotations: str
    augmentations: List[str] = None
    num_workers: Optional[int] = None

@dataclass
class ResourceConfig:
    device: str = "auto"
    memory_config: str = "max_split_size_mb:32"
    pin_memory: bool = False
    persistent_workers: bool = False

class ConfigValidator:
    @staticmethod
    def validate_paths(config: DataConfig) -> None:
        paths = [
            Path(config.train_dir),
            Path(config.val_dir),
            Path(config.train_annotations),
            Path(config.val_annotations)
        ]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

    @staticmethod
    def validate_model_config(config: ModelConfig) -> None:
        if config.num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if config.score_thresh < 0 or config.score_thresh > 1:
            raise ValueError("score_thresh must be between 0 and 1")

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def create_configs(config_dict: dict) -> Tuple[ModelConfig, TrainingConfig, DataConfig, ResourceConfig]:
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        data_config = DataConfig(**config_dict['data'])
        resource_config = ResourceConfig(**config_dict.get('resources', {}))
        
        ConfigValidator.validate_paths(data_config)
        ConfigValidator.validate_model_config(model_config)
        
        return model_config, training_config, data_config, resource_config