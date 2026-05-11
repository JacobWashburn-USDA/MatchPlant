import torch
import torchvision
from PIL import Image, ImageDraw
import json
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import sys

TRAINER_DIR = Path(__file__).resolve().parents[1] / "6-1_obj_det_trainer"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from train import (MaizeDatasetCOCO, get_transform, build_model, 
                  configure_device_and_resources, collate_fn, COCOeval)

def load_model(model_path, device):
    model = build_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def calculate_iou(box1, box2):
    # Calculate IoU between two boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (box1_area + box2_area - intersection)

def analyze_detection_performance(pred_boxes, pred_scores, gt_boxes, iou_thresholds=[0.5, 0.75]):
    """Detailed analysis of detection performance"""
    results = {}
    
    for iou_thresh in iou_thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = len(gt_boxes)
        
        # Track matched ground truth boxes
        matched_gt = set()
        
        # Sort predictions by confidence
        conf_order = torch.argsort(pred_scores, descending=True)
        sorted_boxes = pred_boxes[conf_order]
        sorted_scores = pred_scores[conf_order]
        
        for pred_box, score in zip(sorted_boxes, sorted_scores):
            if score < 0.5:  # Skip low confidence predictions
                continue
                
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou > iou_thresh and best_gt_idx not in matched_gt:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                false_negatives -= 1
            else:
                false_positives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[f'iou_{iou_thresh}'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    return results

def analyze_size_performance(pred_boxes, pred_scores, gt_boxes):
    """Analyze performance based on object size"""
    size_ranges = {
        'small': (0, 32*32),
        'medium': (32*32, 96*96),
        'large': (96*96, float('inf'))
    }
    
    size_results = {size: {'tp': 0, 'fp': 0, 'fn': 0} for size in size_ranges}
    
    # Analyze ground truth boxes by size
    for gt_box in gt_boxes:
        area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        for size, (min_area, max_area) in size_ranges.items():
            if min_area <= area < max_area:
                size_results[size]['fn'] += 1
    
    # Match predictions to ground truth
    for pred_box, score in zip(pred_boxes, pred_scores):
        if score < 0.5:
            continue
            
        area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        matched = False
        
        for gt_box in gt_boxes:
            if calculate_iou(pred_box, gt_box) > 0.5:
                matched = True
                for size, (min_area, max_area) in size_ranges.items():
                    if min_area <= area < max_area:
                        size_results[size]['tp'] += 1
                        size_results[size]['fn'] -= 1
                break
        
        if not matched:
            for size, (min_area, max_area) in size_ranges.items():
                if min_area <= area < max_area:
                    size_results[size]['fp'] += 1
    
    # Calculate metrics for each size
    for size in size_results:
        tp = size_results[size]['tp']
        fp = size_results[size]['fp']
        fn = size_results[size]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        size_results[size].update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    return size_results

def plot_confidence_distribution(all_scores, save_path):
    """Plot distribution of confidence scores"""
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, range=(0, 1))
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Detection Confidence Scores')
    plt.savefig(save_path)
    plt.close()

def create_confusion_matrix(results, save_path):
    """Create and plot confusion matrix"""
    tp = results['iou_0.5']['true_positives']
    fp = results['iou_0.5']['false_positives']
    fn = results['iou_0.5']['false_negatives']
    
    cm = np.array([[tp, fp], [fn, 0]])  # Basic 2x2 confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, str(cm[row, col]), ha='center', va='center')
    plt.xticks([0, 1], ['TP', 'FP'])
    plt.yticks([0, 1], ['FN', ''])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Detection Confusion Matrix (IoU > 0.5)')
    plt.savefig(save_path)
    plt.close()

def predictions_to_coco(predictions, targets, score_threshold=0.05):
    coco_predictions = []
    for pred, target in zip(predictions, targets):
        image_id = int(target["image_id"].item())
        for box, score, label in zip(pred["boxes"].cpu(), pred["scores"].cpu(), pred["labels"].cpu()):
            if float(score) < score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            coco_predictions.append({
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score.item()),
            })
    return coco_predictions

def evaluate_coco_metrics(coco_gt, coco_predictions):
    if COCOeval is None or coco_gt is None:
        raise RuntimeError("pycocotools is required to compute COCO AP metrics")
    if not coco_predictions:
        return [0.0] * 12

    coco_dt = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def aggregate_detection_metrics(all_results, iou_key="iou_0.5"):
    tp = sum(r[iou_key]["true_positives"] for r in all_results)
    fp = sum(r[iou_key]["false_positives"] for r in all_results)
    fn = sum(r[iou_key]["false_negatives"] for r in all_results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }

def write_summary_csv(summary, save_path):
    scalar_keys = [
        "checkpoint",
        "data_dir",
        "annotation_file",
        "AP_50",
        "mAP_50_95",
        "precision",
        "recall",
        "f1_score",
        "mean_inference_time",
        "total_time",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerow({key: summary.get(key) for key in scalar_keys})

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained maize detector.")
    parser.add_argument("--data-dir", default="data/test", help="Directory containing test tiles.")
    parser.add_argument("--annotation-file", default="annotations/test.json", help="COCO annotation JSON for the test split.")
    parser.add_argument("--checkpoint", "--model", default="checkpoints/best_model.pt", help="Model checkpoint to evaluate.")
    parser.add_argument("--results-dir", default="paper_results", help="Directory for output metrics and plots.")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    device, _ = configure_device_and_resources()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Load test dataset and model
    dataset_test = MaizeDatasetCOCO(
        Path(args.data_dir),
        Path(args.annotation_file),
        transforms=get_transform(train=False)
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    model = load_model(args.checkpoint, device)
    
    # Collect results
    all_results = []
    all_scores = []
    all_size_results = []
    processing_times = []
    coco_predictions = []
    
    for i, (images, targets) in enumerate(test_loader):
        images = list(img.to(device) for img in images)
        
        # Measure inference time
        start_inference = time.time()
        with torch.no_grad():
            predictions = model(images)
        inference_time = time.time() - start_inference
        processing_times.append(inference_time)
        
        for pred, target in zip(predictions, targets):
            # Performance analysis
            results = analyze_detection_performance(
                pred['boxes'].cpu(),
                pred['scores'].cpu(),
                target['boxes'].cpu()
            )
            all_results.append(results)
            
            # Size analysis
            size_results = analyze_size_performance(
                pred['boxes'].cpu(),
                pred['scores'].cpu(),
                target['boxes'].cpu()
            )
            all_size_results.append(size_results)
            
            # Collect confidence scores
            all_scores.extend(pred['scores'].cpu().numpy())

        coco_predictions.extend(predictions_to_coco(predictions, targets))
    
    # Aggregate results
    coco_stats = evaluate_coco_metrics(dataset_test.coco, coco_predictions)
    aggregate_iou_05 = aggregate_detection_metrics(all_results, "iou_0.5")
    aggregate_iou_075 = aggregate_detection_metrics(all_results, "iou_0.75")

    final_results = {
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'annotation_file': args.annotation_file,
        'AP_50': float(coco_stats[1]),       # COCOeval stats[1]
        'mAP_50_95': float(coco_stats[0]),   # COCOeval stats[0]
        'precision': float(aggregate_iou_05['precision']),
        'recall': float(aggregate_iou_05['recall']),
        'f1_score': float(aggregate_iou_05['f1_score']),
        'performance': {
            'aggregate_iou_0.5': aggregate_iou_05,
            'aggregate_iou_0.75': aggregate_iou_075,
            'iou_0.5': {metric: float(np.mean([r['iou_0.5'][metric] for r in all_results]))
                       for metric in ['precision', 'recall', 'f1_score']},
            'iou_0.75': {metric: float(np.mean([r['iou_0.75'][metric] for r in all_results]))
                        for metric in ['precision', 'recall', 'f1_score']}
        },
        'size_performance': {
            size: {metric: float(np.mean([r[size][metric] for r in all_size_results]))
                  for metric in ['precision', 'recall', 'f1_score']}
            for size in ['small', 'medium', 'large']
        },
        'timing': {
            'mean_inference_time': float(np.mean(processing_times)),
            'std_inference_time': float(np.std(processing_times)),
            'total_time': float(time.time() - start_time)
        }
    }
    final_results['mean_inference_time'] = final_results['timing']['mean_inference_time']
    final_results['total_time'] = final_results['timing']['total_time']
    
    # Generate plots
    plot_confidence_distribution(all_scores, results_dir / 'confidence_distribution.png')
    create_confusion_matrix(all_results[0], results_dir / 'confusion_matrix.png')
    
    # Save results
    with open(results_dir / 'coco_predictions.json', 'w') as f:
        json.dump(coco_predictions, f, indent=2)
    with open(results_dir / 'paper_metrics.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    with open(results_dir / 'test_summary.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    write_summary_csv(final_results, results_dir / 'test_summary.csv')
    
    # Print summary
    print("\nResults Summary for Academic Paper:")
    print(f"COCO mAP@0.5:0.95: {final_results['mAP_50_95']:.3f}")
    print(f"COCO AP@0.5: {final_results['AP_50']:.3f}")
    print("\nPerformance Metrics:")
    print(f"IoU@0.5 - Precision: {final_results['precision']:.3f}")
    print(f"IoU@0.5 - Recall: {final_results['recall']:.3f}")
    print(f"IoU@0.5 - F1 Score: {final_results['f1_score']:.3f}")
    
    print("\nSize-based Performance:")
    for size in ['small', 'medium', 'large']:
        print(f"\n{size.capitalize()} Objects:")
        print(f"Precision: {final_results['size_performance'][size]['precision']:.3f}")
        print(f"Recall: {final_results['size_performance'][size]['recall']:.3f}")
        print(f"F1 Score: {final_results['size_performance'][size]['f1_score']:.3f}")
    
    print("\nTiming Information:")
    print(f"Mean inference time: {final_results['timing']['mean_inference_time']*1000:.2f}ms")
    print(f"Total processing time: {final_results['timing']['total_time']:.2f}s")

if __name__ == "__main__":
    main()
