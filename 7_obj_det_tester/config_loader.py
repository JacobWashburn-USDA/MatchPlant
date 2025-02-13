"""
Script Name: test_config_loader.py
Purpose: To manage configuration for object detection testing
Author: Worasit Sangjan
Date: 12 February 2025
"""

import yaml
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

@dataclass
class VisualizationConfig:
    """Visualization settings for test results"""
    confidence_threshold: float = 0.5
    pred_box_color: str = 'red'
    gt_box_color: str = 'blue'
    box_width: int = 2
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 100
    save_visualizations: bool = True

@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics"""
    iou_thresholds: List[float] = (0.5, 0.75)
    confidence_threshold: float = 0.5
    size_ranges: Dict[str, Tuple[int, Optional[int]]] = None

    def __post_init__(self):
        if self.size_ranges is None:
            self.size_ranges = {
                'small': (0, 32*32),
                'medium': (32*32, 96*96),
                'large': (96*96, None)
            }
        else:
            # Convert lists from YAML to tuples
            self.size_ranges = {
                size: (range_[0], range_[1] if range_[1] is not None else None)
                for size, range_ in self.size_ranges.items()
            }
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not all(0 <= iou <= 1 for iou in self.iou_thresholds):
            raise ValueError("All IoU thresholds must be between 0 and 1")

@dataclass
class DataConfig:
    """Test dataset configuration"""
    test_dir: str
    test_annotations: str
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")

@dataclass
class OutputConfig:
    """Configuration for test output and results"""
    results_dir: str = './paper_results'
    test_results_dir: str = './test_results'
    save_individual_results: bool = True
    save_aggregate_results: bool = True
    save_visualizations: bool = True

@dataclass
class TestConfig:
    """Main test configuration"""
    model_path: str = './checkpoints/best_model.pt'
    num_runs: int = 5
    metrics: MetricsConfig = None
    data: DataConfig = None
    visualization: VisualizationConfig = None
    output: OutputConfig = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = MetricsConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.output is None:
            self.output = OutputConfig()

class ConfigValidator:
    """Validates test configuration settings"""
    
    @staticmethod
    def validate_paths(config: TestConfig) -> None:
        """Validate all required paths exist"""
        paths = [
            Path(config.model_path),
            Path(config.data.test_dir),
            Path(config.data.test_annotations)
        ]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

class TestConfigLoader:
    """Configuration loader for testing"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    @staticmethod
    def create_config(config_dict: Dict[str, Any]) -> TestConfig:
        """Create and validate test configuration object"""
        try:
            metrics_config = MetricsConfig(**config_dict.get('metrics', {}))
            data_config = DataConfig(**config_dict['data'])
            viz_config = VisualizationConfig(**config_dict.get('visualization', {}))
            output_config = OutputConfig(**config_dict.get('output', {}))
            
            test_config = TestConfig(
                model_path=config_dict.get('model_path', './checkpoints/best_model.pt'),
                num_runs=config_dict.get('num_runs', 5),
                metrics=metrics_config,
                data=data_config,
                visualization=viz_config,
                output=output_config
            )
            
            ConfigValidator.validate_paths(test_config)
            
            return test_config
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration section: {e}")
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameter: {e}")

    @staticmethod
    def save_config(config_dict: Dict[str, Any], output_path: str) -> None:
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)