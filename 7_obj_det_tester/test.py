"""
Script Name: test.py
Purpose: Test and evaluates a Faster R-CNN model for object detection using PyTorch, 
        with COCO format dataset handling and evaluation metrics
Authors: Worasit Sangjan
Date: 4 May 2025 
Version: 1.3
"""

import torch
import torchvision
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time
import seaborn as sns
from test_config_loader import TestConfigLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from train import (MaizeDatasetCOCO, get_transform, build_model, 
                  configure_device_and_resources, collate_fn)
from prediction_coco_converter import create_prediction_annotations

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_model(model_path, device):
    model = build_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

class COCOEvaluator:
    """COCO evaluator - using the same approach as in train.py"""
    def __init__(self, coco_gt, config):
        self.coco_gt = coco_gt
        self.config = config
        self.results = []
        # Get only image IDs that have annotations
        annotation_imgs = set()
        for ann in self.coco_gt.anns.values():
            annotation_imgs.add(ann['image_id'])
        self.img_ids = sorted(list(annotation_imgs))
        print(f"\nInitialized evaluator with {len(self.img_ids)} test images")
        
        # Save all confidence scores for later analysis
        self.all_scores = []

    def update(self, predictions, targets):
        for pred, target in zip(predictions, targets):
            image_id = target["image_id"].item()
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]
            
            # Collect all confidence scores
            self.all_scores.extend(scores.cpu().numpy())
            
            # Convert predictions to COCO format
            for box, score, label in zip(boxes, scores, labels):
                if score > self.config.metrics.confidence_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    
                    self.results.append({
                        "image_id": image_id,
                        "category_id": label.item(),
                        "bbox": [x1, y1, width, height],
                        "score": score.item()
                    })

    def evaluate(self):
        if not self.results:
            return {}
            
        # Create COCO results object
        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = self.img_ids
        
        # Get overall mAP metrics
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Store comprehensive mAP metrics
        metrics = {
            'AP_IoU=0.50:0.95': coco_eval.stats[0],  # mAP averaged over IoUs
            'AP_IoU=0.50': coco_eval.stats[1],       # mAP at IoU=0.50
            'AP_IoU=0.75': coco_eval.stats[2],       # mAP at IoU=0.75
            'AP_small': coco_eval.stats[3],          # mAP for small objects
            'AP_medium': coco_eval.stats[4],         # mAP for medium objects
            'AP_large': coco_eval.stats[5],          # mAP for large objects
            'AR_maxDets=1': coco_eval.stats[6],      # AR given 1 detection per image
            'AR_maxDets=10': coco_eval.stats[7],     # AR given 10 detections per image
            'AR_maxDets=100': coco_eval.stats[8],    # AR given 100 detections per image
            'AR_small': coco_eval.stats[9],          # AR for small objects
            'AR_medium': coco_eval.stats[10],        # AR for medium objects
            'AR_large': coco_eval.stats[11]          # AR for large objects
        }
        
        # Evaluate by size using size ranges from config
        size_performance = {}
        for size, (min_area, max_area) in self.config.metrics.size_ranges.items():
            # Set area range for current size category
            if max_area is None:
                max_area = float('inf')
            coco_eval.params.areaRng = [[min_area, max_area]]
            
            # Set IoU threshold to 0.5 to align with common reporting practices
            coco_eval.params.iouThrs = [0.5]
                
            coco_eval.evaluate()
            coco_eval.accumulate()
            
            # Get metrics for this size category at IoU=0.5
            precision = coco_eval.eval['precision']
            precision_value = float(np.mean(precision[0, :, 0, 0, -1]))
            
            recall = coco_eval.eval['recall']
            recall_value = float(np.mean(recall[0, :, 0]))
            
            f1_value = 2 * (precision_value * recall_value) / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0
            
            size_performance[size] = {
                'precision': precision_value,
                'recall': recall_value,
                'f1_score': f1_value
            }
        
        return metrics, size_performance

def plot_confidence_distribution(all_scores, save_path, config):
    plt.figure(figsize=config.visualization.figure_size, dpi=config.visualization.dpi)
    plt.hist(all_scores, bins=50, range=(0, 1))
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Detection Confidence Scores')
    plt.savefig(save_path, dpi=config.visualization.dpi)
    plt.close()

def visualize_test_results(image, pred_boxes, pred_scores, gt_boxes, target, output_dir, config, coco_gt):
    """Visualize detection results with predicted and ground truth boxes"""
    if isinstance(image, torch.Tensor):
        image = torchvision.transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(image)
    
    # Draw predicted boxes
    for box, score in zip(pred_boxes, pred_scores):
        if score > config.visualization.confidence_threshold:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle(
                [x1, y1, x2, y2], 
                outline=config.visualization.pred_box_color,
                width=config.visualization.box_width
            )
            draw.text((x1, y1-10), f'{score:.2f}', 
                     fill=config.visualization.pred_box_color)
    
    # Draw ground truth boxes
    for box in gt_boxes:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle(
            [x1, y1, x2, y2], 
            outline=config.visualization.gt_box_color,
            width=config.visualization.box_width
        )
    
    # Get original filename from COCO dataset using the passed coco_gt object
    image_info = coco_gt.loadImgs(target['image_id'].item())[0]
    original_filename = image_info['file_name']
    
    # Create save path with original filename but PNG extension
    save_name = Path(original_filename).stem + '.png'
    final_save_path = output_dir / save_name
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save visualization
    image.save(str(final_save_path))

def run_single_test(config=None, run_dir=None):
    """Run a single test iteration and return results"""
    if config is None:
        config_loader = TestConfigLoader()
        config = config_loader.create_config(
            config_loader.load_config('test_config.yaml')
        )
    
    start_time = time.time()
    device, _ = configure_device_and_resources()
    
    # Use config parameters or run-specific directory if provided
    results_dir = Path(run_dir) if run_dir else Path(config.output.results_dir)
    results_dir.mkdir(exist_ok=True)
    test_results_dir = results_dir / 'images' 
    test_results_dir.mkdir(exist_ok=True)
    
    # Load test dataset using config
    dataset_test = MaizeDatasetCOCO(
        Path(config.data.test_dir),
        Path(config.data.test_annotations),
        transforms=get_transform(train=False)
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # Initialize evaluator
    coco_gt = COCO(config.data.test_annotations)
    evaluator = COCOEvaluator(coco_gt, config)

    model = load_model(config.model_path, device)

    # Create prediction annotations for analysis
    create_prediction_annotations(
        test_loader=test_loader,
        model=model,
        test_annotation_path=config.data.test_annotations,
        output_dir=results_dir,
        confidence_threshold=config.visualization.confidence_threshold,
        device=device
    )    

    # Process images and collect timings
    processing_times = []
    
    for i, (images, targets) in enumerate(test_loader):
        images = list(img.to(device) for img in images)
        
        # Measure inference time
        start_inference = time.time()
        with torch.no_grad():
            predictions = model(images)
        inference_time = time.time() - start_inference
        processing_times.append(inference_time)
        
        # Update evaluator
        evaluator.update(predictions, targets)
        
        # Visualize results if enabled
        if config.visualization.save_visualizations:
            for idx, (pred, target) in enumerate(zip(predictions, targets)):
                visualize_test_results(
                    images[idx].cpu(),
                    pred['boxes'].cpu(),
                    pred['scores'].cpu(),
                    target['boxes'].cpu(),
                    target,  
                    test_results_dir,
                    config,
                    coco_gt
                )
    
    # Get performance metrics using COCO evaluation
    metrics, size_performance = evaluator.evaluate()
    
    # Return results
    return {
        'coco_metrics': metrics,  
        'size_performance': size_performance,
        'timing': {
            'mean_inference_time': np.mean(processing_times),
            'std_inference_time': np.std(processing_times),
            'total_time': time.time() - start_time
        },
        'all_scores': evaluator.all_scores
    }

def run_multiple_tests(num_runs=None):
    """Run multiple test iterations and compute statistics"""
    config_loader = TestConfigLoader()
    config = config_loader.create_config(
        config_loader.load_config('test_config.yaml')
    )
    
    if num_runs is None:
        num_runs = config.num_runs
        
    all_runs_results = []
    
    for run in range(num_runs):
        print(f"\nRunning test {run + 1}/{num_runs}")
        
        # Create directory for this run
        run_dir = Path(config.output.results_dir) / f'run_{run + 1}'
        run_dir.mkdir(exist_ok=True, parents=True)
        
        # Run test with run-specific directory
        results = run_single_test(config, run_dir=run_dir)
        all_runs_results.append(results)
        
        if config.output.save_individual_results:
            # Generate confidence distribution plot
            plot_confidence_distribution(
                results['all_scores'],
                run_dir / 'confidence_distribution.png',
                config
            )
        
        # Save individual run results
        with open(run_dir / f'metrics_run_{run + 1}.json', 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # Calculate statistics
    stats = calculate_statistics(all_runs_results)
    
    # Save aggregate results in main results directory
    aggregate_path = Path(config.output.results_dir) / 'aggregate_results.json'
    with open(aggregate_path, 'w') as f:
        json.dump(stats, f, indent=4, cls=NumpyEncoder)
    
    # Print summary
    print_summary(stats)
    
    return stats

def calculate_statistics(all_runs_results):
    """Calculate statistics across multiple runs"""
    stats = {
        'mean': {},
        'std': {},
        'min': {},
        'max': {},
        '95_confidence': {}
    }
    
    # Add COCO metrics statistics
    stats['mean']['coco_metrics'] = {}
    stats['std']['coco_metrics'] = {}
    stats['min']['coco_metrics'] = {}
    stats['max']['coco_metrics'] = {}
    stats['95_confidence']['coco_metrics'] = {}
    
    # Calculate statistics for COCO metrics
    coco_keys = all_runs_results[0]['coco_metrics'].keys()
    for key in coco_keys:
        values = [run['coco_metrics'][key] for run in all_runs_results]
        stats['mean']['coco_metrics'][key] = np.mean(values)
        stats['std']['coco_metrics'][key] = np.std(values)
        stats['min']['coco_metrics'][key] = np.min(values)
        stats['max']['coco_metrics'][key] = np.max(values)
        stats['95_confidence']['coco_metrics'][key] = 1.96 * stats['std']['coco_metrics'][key] / np.sqrt(len(all_runs_results))
    
    # Size-based performance statistics
    stats['mean']['size_performance'] = {}
    stats['std']['size_performance'] = {}
    stats['min']['size_performance'] = {}
    stats['max']['size_performance'] = {}
    stats['95_confidence']['size_performance'] = {}
    
    categories = ['small', 'medium', 'large']
    
    for category in categories:
        stats['mean']['size_performance'][category] = {}
        stats['std']['size_performance'][category] = {}
        stats['min']['size_performance'][category] = {}
        stats['max']['size_performance'][category] = {}
        stats['95_confidence']['size_performance'][category] = {}
        
        for metric in ['precision', 'recall', 'f1_score']:
            values = [run['size_performance'][category][metric] for run in all_runs_results]
            
            stats['mean']['size_performance'][category][metric] = np.mean(values)
            stats['std']['size_performance'][category][metric] = np.std(values)
            stats['min']['size_performance'][category][metric] = np.min(values)
            stats['max']['size_performance'][category][metric] = np.max(values)
            
            confidence = 1.96 * stats['std']['size_performance'][category][metric] / np.sqrt(len(all_runs_results))
            stats['95_confidence']['size_performance'][category][metric] = confidence
    
    return stats

def print_summary(stats):
    """Print summary of results"""
    print("\nAggregate Results Summary:")
    
    print("\nMean Average Precision Metrics:")
    for metric in ['AP_IoU=0.50:0.95', 'AP_IoU=0.50', 'AP_IoU=0.75']:
        mean = stats['mean']['coco_metrics'][metric]
        conf = stats['95_confidence']['coco_metrics'][metric]
        std = stats['std']['coco_metrics'][metric]
        min_val = stats['min']['coco_metrics'][metric]
        max_val = stats['max']['coco_metrics'][metric]
        print(f"{metric}: {mean:.3f} ± {conf:.3f} "
              f"(std: {std:.3f}, min: {min_val:.3f}, max: {max_val:.3f})")
    
    print("\nSize-based AP Metrics:")
    for metric in ['AP_small', 'AP_medium', 'AP_large']:
        mean = stats['mean']['coco_metrics'][metric]
        conf = stats['95_confidence']['coco_metrics'][metric]
        std = stats['std']['coco_metrics'][metric]
        min_val = stats['min']['coco_metrics'][metric]
        max_val = stats['max']['coco_metrics'][metric]
        print(f"{metric}: {mean:.3f} ± {conf:.3f} "
              f"(std: {std:.3f}, min: {min_val:.3f}, max: {max_val:.3f})")
    
    print("\nAverage Recall Metrics:")
    for metric in ['AR_maxDets=1', 'AR_maxDets=10', 'AR_maxDets=100']:
        mean = stats['mean']['coco_metrics'][metric]
        conf = stats['95_confidence']['coco_metrics'][metric]
        std = stats['std']['coco_metrics'][metric]
        min_val = stats['min']['coco_metrics'][metric]
        max_val = stats['max']['coco_metrics'][metric]
        print(f"{metric}: {mean:.3f} ± {conf:.3f} "
              f"(std: {std:.3f}, min: {min_val:.3f}, max: {max_val:.3f})")
    
    print("\nSize-based Performance (IoU=0.5):")
    for size in ['small', 'medium', 'large']:
        print(f"\n{size.capitalize()} Objects:")
        for metric in ['precision', 'recall', 'f1_score']:
            mean = stats['mean']['size_performance'][size][metric]
            conf = stats['95_confidence']['size_performance'][size][metric]
            std = stats['std']['size_performance'][size][metric]
            min_val = stats['min']['size_performance'][size][metric]
            max_val = stats['max']['size_performance'][size][metric]
            print(f"{metric.capitalize()}: {mean:.3f} ± {conf:.3f} "
                  f"(std: {std:.3f}, min: {min_val:.3f}, max: {max_val:.3f})")

if __name__ == "__main__":
    stats = run_multiple_tests()