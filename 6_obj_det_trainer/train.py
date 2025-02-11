"""
Script Name: train.py
Purpose: Trains and evaluates a Faster R-CNN model for maize object detection using PyTorch, 
        with COCO format dataset handling and evaluation metrics
Authors: Worasit Sangjan and Piyush Pandey
Date: 11 February 2025 
Version: 1.3
"""

import os, platform, multiprocessing, psutil
import torch, torchvision
import torch.utils.data
from PIL import Image
import json
import datetime
from config_loader import ConfigLoader 
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def configure_device_and_resources():
    system = platform.system()
    device = torch.device("cuda" if torch.cuda.is_available() else 
             "mps" if system == "Darwin" and torch.backends.mps.is_available() else 
             "cpu")
    
    if system == "Darwin":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['TORCH_SHM_DISABLE'] = '1'
        num_workers = 0  # Reduced workers for MacOS stability
    else:
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        num_workers = min(int(total_ram_gb // 16), multiprocessing.cpu_count())
    
    print(f"Running on: {system}")
    print(f"Using device: {device}")
    print(f"Workers: {num_workers}")
    
    return device, num_workers

def collate_fn(batch):
    return tuple(zip(*batch))

class MaizeDatasetCOCO(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotation_file, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        
        # Load annotations
        print(f"Loading dataset from {annotation_file}")
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.coco = COCO(annotation_file)
        
        # Create mapping of image IDs to annotations
        self.img_to_anns = {}
        valid_image_ids = set()
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            valid_image_ids.add(img_id)
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Filter images to only those with annotations
        self.images = [img for img in self.coco_data['images'] 
                      if img['id'] in valid_image_ids]
        
        # Print dataset statistics
        print(f"Dataset contains {len(self.images)} images with annotations")
        print(f"Image IDs in dataset: {sorted([img['id'] for img in self.images])}")

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load and process image
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        anns = self.img_to_anns[img_id]
        
        boxes = []
        labels = []
        
        # Process annotations
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }
        
        if self.transforms:
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return len(self.images)

class DetectionTransform:
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target

class DetectionTransformTrain(DetectionTransform):
    def __init__(self, aug_config):
        self.aug_config = aug_config
        
    def __call__(self, image, target):
        if not isinstance(target, dict) or 'boxes' not in target:
            raise ValueError("Target must be a dict with 'boxes' key")

        image = T.functional.to_tensor(image)
        
        # Color jittering using config values
        if torch.rand(1) < 0.5:
            color_jitter = T.ColorJitter(
                brightness=self.aug_config.brightness,
                contrast=self.aug_config.contrast,
                saturation=self.aug_config.saturation
            )
            image = color_jitter(image)
        
        # Random horizontal flip using config probability
        if torch.rand(1) < self.aug_config.horizontal_flip_prob:
            image = T.functional.hflip(image)
            if len(target["boxes"]):
                bbox = target["boxes"]
                bbox[:, [0, 2]] = image.shape[-1] - bbox[:, [2, 0]]
                target["boxes"] = bbox
        
        return image, target

def get_transform(train, aug_config=None):
    if train:
        return DetectionTransformTrain(aug_config)
    return DetectionTransform()

def build_model(model_config):
    # Custom anchor generator using config values
    anchor_generator = AnchorGenerator(
        sizes=model_config.anchor_sizes,
        aspect_ratios=model_config.anchor_ratios
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_detections_per_img=model_config.box_detections_per_img,
        min_size=model_config.min_size,
        max_size=model_config.max_size,
        box_score_thresh=model_config.box_score_thresh,
        box_nms_thresh=model_config.box_nms_thresh,
        rpn_pre_nms_top_n_train=model_config.rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=model_config.rpn_post_nms_top_n_train,
        rpn_fg_iou_thresh=model_config.rpn_fg_iou_thresh,
        rpn_bg_iou_thresh=model_config.rpn_bg_iou_thresh,
        rpn_post_nms_top_n_test=model_config.rpn_post_nms_top_n_test,
        box_fg_iou_thresh=model_config.box_fg_iou_thresh,
        box_bg_iou_thresh=model_config.box_bg_iou_thresh,
        box_batch_size_per_image=model_config.box_batch_size_per_image,
        box_positive_fraction=model_config.box_positive_fraction,
        anchor_generator=anchor_generator
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, model_config.num_classes)
    
    return model

class COCOEvaluator:
    def __init__(self, coco_gt, iou_types):
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        # Get only the image IDs that actually have annotations
        annotation_imgs = set()
        for ann in self.coco_gt.anns.values():
            annotation_imgs.add(ann['image_id'])
        self.img_ids = sorted(list(annotation_imgs))
        
        self.results = []
        print(f"\nInitialized evaluator with {len(self.img_ids)} validation images")
        print(f"Validation image IDs: {self.img_ids}")

    def update(self, predictions):
        for prediction in predictions:
            image_id = prediction["image_id"].item()
            
            # Skip if image not in validation set
            if image_id not in self.img_ids:
                print(f"Skipping predictions for image {image_id} - not in validation set")
                continue
                
            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]
            
            # Apply confidence threshold and convert boxes to COCO format
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.05:  # Apply confidence threshold
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1 
                    height = y2 - y1
                    
                    self.results.append({
                        "image_id": image_id,
                        "category_id": label.item(),
                        "bbox": [x1, y1, width, height],
                        "score": score.item()
                    })
            
            print(f"Processed image {image_id} with {len(boxes)} detections")
    
    def summarize(self):
        print("\nEvaluation Summary:")
        if not self.results:
            print("No predictions to evaluate")
            return [0.0] * 12

        try:
            coco_dt = self.coco_gt.loadRes(self.results)
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.params.imgIds = self.img_ids
            
            print(f"Evaluating on {len(self.img_ids)} images")
            print(f"Image IDs: {self.img_ids}")
            
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            return coco_eval.stats
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return [0.0] * 12

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    try:
        for i, (images, targets) in enumerate(data_loader):
            # Clear cache at start of each iteration
            torch.cuda.empty_cache()
            
            # Move data to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]     

            # Forward pass with mixed precision
            try:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
            except Exception as e:
                print(f"Forward pass error at iteration {i}: {e}")
                continue

            # Store loss values before cleanup
            loss_value = losses.item()
            loss_classifier_value = loss_dict['loss_classifier'].item()
            loss_box_reg_value = loss_dict['loss_box_reg'].item()
            loss_objectness_value = loss_dict['loss_objectness'].item()
            loss_rpn_box_reg_value = loss_dict['loss_rpn_box_reg'].item()
            
            # Backward pass
            total_loss += loss_value
            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Cleanup
            del losses
            del loss_dict
            del images
            torch.cuda.empty_cache()

            # Periodic garbage collection
            if i % 10 == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Print progress
                lr = optimizer.param_groups[0]["lr"]
                num_boxes = sum(len(t["boxes"]) for t in targets)
                print(f"Epoch: [{epoch}][{i}/{len(data_loader)}] "
                    f"boxes: {num_boxes} "
                    f"lr: {lr:.6f} "
                    f"loss: {loss_value:.4f} "
                    f"loss_classifier: {loss_classifier_value:.4f} "
                    f"loss_box_reg: {loss_box_reg_value:.4f} "
                    f"loss_objectness: {loss_objectness_value:.4f} "
                    f"loss_rpn_box_reg: {loss_rpn_box_reg_value:.4f}")

            # Clear target references
            del targets
            
    except Exception as e:
        print(f"Training iteration error: {e}")
        import traceback
        traceback.print_exc()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device, epoch=None, train_loss=None):
    """Evaluate the model on the validation dataset and save detailed results."""
    model.eval()
    model.roi_heads.score_thresh = 0.05
    evaluator = COCOEvaluator(data_loader.dataset.coco, ["bbox"])
    
    # Print validation set info at start
    print(f"\nStarting evaluation on {len(evaluator.img_ids)} validation images")
    print(f"Valid image IDs: {sorted(list(evaluator.img_ids))}")
    
    skipped_ids = set()
    processed_ids = set()
    
    with torch.no_grad():
        for images, targets in data_loader:
            # Clear cache before processing batch
            torch.cuda.empty_cache()
            
            images = list(img.to(device) for img in images)
            outputs = model(images)

            processed_outputs = []
            for output, target in zip(outputs, targets):
                image_id = target["image_id"].item()
                
                if image_id not in evaluator.img_ids:
                    skipped_ids.add(image_id)
                    continue
                    
                processed_ids.add(image_id)
                
                # Apply score threshold
                keep = output['scores'] > 0.05
                filtered_output = {
                    'boxes': output['boxes'][keep],
                    'scores': output['scores'][keep],
                    'labels': output['labels'][keep],
                    'image_id': target['image_id']
                }
                processed_outputs.append(filtered_output)
            
            evaluator.update(processed_outputs)
            
            # Clean up GPU memory
            del images
            del outputs
            torch.cuda.empty_cache()
    
    # Print summary at end
    print(f"\nEvaluation Summary:")
    print(f"Processed {len(processed_ids)} images: {sorted(list(processed_ids))}")
    if skipped_ids:
        print(f"Skipped {len(skipped_ids)} images: {sorted(list(skipped_ids))}")
    
    # Get evaluation stats
    stats = evaluator.summarize()
    
    # Setup results directory
    results_dir = Path('./validation_results')
    results_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create detailed results dictionary
    results = {
        'timestamp': timestamp,
        'epoch': epoch,
        'train_loss': train_loss,
        'stats': {
            'AP_IoU=0.50:0.95': stats[0],
            'AP_IoU=0.50': stats[1],
            'AP_IoU=0.75': stats[2],
            'AP_small': stats[3],
            'AP_medium': stats[4],
            'AP_large': stats[5],
            'AR_IoU=0.50:0.95_maxDets=1': stats[6],
            'AR_IoU=0.50:0.95_maxDets=10': stats[7],
            'AR_IoU=0.50:0.95_maxDets=100': stats[8],
            'AR_small': stats[9],
            'AR_medium': stats[10],
            'AR_large': stats[11]
        },
        'processed_images': len(processed_ids),
        'skipped_images': len(skipped_ids),
        'processed_image_ids': sorted(list(processed_ids)),
        'skipped_image_ids': sorted(list(skipped_ids))
    }
    
    # Save detailed results for this evaluation
    if epoch is not None:
        filename = f'validation_results_epoch_{epoch}_{timestamp}.json'
    else:
        filename = f'validation_results_{timestamp}.json'
        
    with open(results_dir / filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed validation results saved to: {filename}")
    
    # Update the training summary file if epoch is provided
    if epoch is not None:
        summary = {
            'epoch': epoch,
            'timestamp': timestamp,
            'train_loss': train_loss,
            'mAP_50': stats[1],      
            'mAP_75': stats[2],      
            'mAP_small': stats[3],     
            'mAP_medium': stats[4],    
            'mAP_large': stats[5],     
            'processed_images': len(processed_ids)
        }
        
        summary_file = results_dir / 'validation_summary.jsonl'
        with open(summary_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
        print(f"Validation summary updated in: validation_summary.jsonl")
    
    return stats

def main():
    # Load configuration
    config_loader = ConfigLoader()
    model_config, training_config, data_config, resource_config = config_loader.create_configs(
        config_loader.load_config('config.yaml')
    )
    
    # Configure environment
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = resource_config.memory_config
    torch.backends.cudnn.benchmark = resource_config.cuda_benchmark
    torch.cuda.empty_cache()

    device, num_workers = configure_device_and_resources()
    if data_config.num_workers is not None:
        num_workers = data_config.num_workers
    
    # Create datasets using config paths
    dataset = MaizeDatasetCOCO(
        Path(data_config.train_dir),
        Path(data_config.train_annotations),
        transforms=get_transform(train=True, aug_config=training_config.augmentation)
    )
    
    dataset_val = MaizeDatasetCOCO(
        Path(data_config.val_dir),
        Path(data_config.val_annotations),
        transforms=get_transform(train=False)
    )
    
    # Create data loaders using config settings
    data_loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers
    )
    
    # Initialize model using config
    model = build_model(model_config)
    model.to(device)
    
    # Optimizer using config settings
    optimizer = torch.optim.SGD(
        [
            {'params': model.backbone.parameters(), 'lr': training_config.optimizer.backbone_lr},
            {'params': model.rpn.parameters(), 'lr': training_config.optimizer.rpn_lr},
            {'params': model.roi_heads.parameters(), 'lr': training_config.optimizer.roi_heads_lr}
        ],
        momentum=training_config.optimizer.momentum,
        weight_decay=training_config.optimizer.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config.optimizer.lr_step_size,
        gamma=training_config.optimizer.lr_gamma
    )
    
    # Setup checkpointing using config
    checkpoint_dir = Path(training_config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop using config epochs and save frequency
    try:
        best_map = 0.0
        for epoch in range(training_config.epochs):
            print(f"\nEpoch {epoch}/{training_config.epochs}")
            print("-" * 20)
            
            train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
            print(f"Train Loss: {train_loss:.4f}")

            val_stats = evaluate(model, data_loader_val, device, epoch=epoch, train_loss=train_loss)
            val_map = val_stats[1]
            print(f"Validation mAP: {val_map:.4f}")
            
            lr_scheduler.step()
            
            if val_map > best_map:
                best_map = val_map
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': best_map,
                }, checkpoint_dir / 'best_model.pt')
            
            if epoch % training_config.save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_map': val_map,
                }, checkpoint_dir / f'model_epoch_{epoch}.pt')
                
    except Exception as e:
        print(f"Training error: {e}")
        torch.save(model.state_dict(), checkpoint_dir / 'emergency_save.pt')

if __name__ == "__main__":
    main()