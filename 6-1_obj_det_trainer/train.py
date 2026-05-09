import argparse
import os, platform, multiprocessing, psutil
import random
import torch, torchvision
import torch.utils.data
from PIL import Image
import json
import numpy as np
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
from torch.utils.data import DataLoader
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ModuleNotFoundError:
    COCO = None
    COCOeval = None

# Constants
NUM_EPOCHS = 150
BATCH_SIZE = 1
NUM_CLASSES = 2  # Background + maize

def set_seed(seed):
    """Seed all training RNGs for the reviewer-response five-run workflow."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        
        self.coco = COCO(annotation_file) if COCO is not None else None
        
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
    def __call__(self, image, target):
        if not isinstance(target, dict) or 'boxes' not in target:
            raise ValueError("Target must be a dict with 'boxes' key")

        # Convert to tensor first    
        image = T.functional.to_tensor(image)
        # Apply horizontal flip with 50% probability
        if torch.rand(1) < 0.5:
            image = T.functional.hflip(image)
            if len(target["boxes"]):
                bbox = target["boxes"]
                bbox[:, [0, 2]] = image.shape[-1] - bbox[:, [2, 0]] 
                target["boxes"] = bbox
                
        return image, target

def get_transform(train):
    if train:
        return DetectionTransformTrain()
    return DetectionTransform() 

def build_model(num_classes, pretrained=True):
    # Custom anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 96, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
        box_detections_per_img=100,
        min_size=980,
        max_size=1240,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=500,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_post_nms_top_n_test=500,
        anchor_generator=anchor_generator  # Pass as a named argument
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Explicitly set evaluation parameters
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 200
    
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
        if COCOeval is None:
            raise RuntimeError("pycocotools is required for COCO evaluation")

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

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, use_amp=True, clear_cache=False):
    model.train()
    total_loss = 0
    use_amp = use_amp and device.type == "cuda"
    
    try:
        for i, (images, targets) in enumerate(data_loader):
            if clear_cache and device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Move data to device
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]     

            # Forward pass with mixed precision
            try:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
            if scaler is not None and scaler.is_enabled():
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Cleanup
            del losses
            del loss_dict
            del images
            if clear_cache and device.type == "cuda":
                torch.cuda.empty_cache()

            # Periodic garbage collection
            if i % 10 == 0:
                if clear_cache:
                    import gc
                    gc.collect()
                    if device.type == "cuda":
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

def evaluate(
    model,
    data_loader,
    device,
    epoch=None,
    train_loss=None,
    results_dir=Path('./validation_results'),
    clear_cache=False,
):
    """
    Evaluate the model on the validation dataset and save detailed results.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation dataset
        device: Device to run evaluation on
        epoch: Current epoch number (optional)
        train_loss: Training loss from current epoch (optional)
    """
    import datetime
    from pathlib import Path
    
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
            if clear_cache and device.type == "cuda":
                torch.cuda.empty_cache()
            
            images = [img.to(device, non_blocking=True) for img in images]
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
            if clear_cache and device.type == "cuda":
                torch.cuda.empty_cache()
    
    # Print summary at end
    print(f"\nEvaluation Summary:")
    print(f"Processed {len(processed_ids)} images: {sorted(list(processed_ids))}")
    if skipped_ids:
        print(f"Skipped {len(skipped_ids)} images: {sorted(list(skipped_ids))}")
    
    # Get evaluation stats
    stats = evaluator.summarize()
    
    # Setup results directory. Seeded runs pass runs/<name>/validation_results here.
    results_dir = Path(results_dir)
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
            'mAP_50_95': stats[0],     # COCO mAP at IoU=0.50:0.95
            'AP_50': stats[1],         # AP at IoU=0.50
            'AP_75': stats[2],         # AP at IoU=0.75
            'mAP_small': stats[3],     # AP for small objects
            'mAP_medium': stats[4],    # AP for medium objects
            'mAP_large': stats[5],     # AP for large objects
            'processed_images': len(processed_ids)
        }
        
        summary_file = results_dir / 'training_summary.jsonl'
        with open(summary_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
        print(f"Training summary updated in: training_summary.jsonl")
    
    return stats

def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for maize detection.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for one independent training run.")
    parser.add_argument("--run_name", default=None, help="Run name. Used as runs/<run_name> when --output_dir is not set.")
    parser.add_argument("--output_dir", default=None, help="Run output directory for checkpoints and validation logs.")
    parser.add_argument("--train-data-dir", default="data/train", help="Fixed training tile directory.")
    parser.add_argument("--val-data-dir", default="data/val", help="Fixed validation tile directory.")
    parser.add_argument("--train-annotation-file", default="annotations/train.json", help="Fixed training COCO JSON.")
    parser.add_argument("--val-annotation-file", default="annotations/val.json", help="Fixed validation COCO JSON.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Images per training batch.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to an automatic value.")
    parser.add_argument("--val-frequency", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many validations without improvement. Disabled by default.",
    )
    parser.add_argument("--lr", type=float, default=0.05, help="SGD learning rate.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision training.")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned-memory DataLoader transfer on CUDA.")
    parser.add_argument("--clear-cache-each-step", action="store_true", help="Clear CUDA cache each step. Slower, but can help debug memory pressure.")
    parser.add_argument(
        "--best-metric",
        choices=["ap50", "map50_95"],
        default="ap50",
        help="Validation metric used for best checkpoint selection. Default preserves current AP@0.5 behavior.",
    )
    return parser.parse_args()

def resolve_run_dirs(args):
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.run_name:
        output_dir = Path("runs") / args.run_name
    else:
        output_dir = None

    if output_dir is None:
        return Path("./checkpoints"), Path("./validation_results")

    checkpoint_dir = output_dir / "checkpoints"
    validation_dir = output_dir / "validation_results"
    existing_outputs = [
        checkpoint_dir / "best_model.pt",
        validation_dir / "training_summary.jsonl",
    ]
    if any(path.exists() for path in existing_outputs):
        raise FileExistsError(
            f"{output_dir} already contains run outputs. Choose a new --output_dir/--run_name to avoid overwriting."
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, validation_dir

def main():
    args = parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if args.seed is not None:
        print(f"Using seeded reviewer-response training run: seed={args.seed}")
        set_seed(args.seed)
    else:
        torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()  # Clear cache before training

    #os.environ['MallocStackLogging'] = '0'
    device, num_workers = configure_device_and_resources()
    num_workers = args.num_workers if args.num_workers is not None else num_workers
    pin_memory = device.type == "cuda" and not args.no_pin_memory
    use_amp = device.type == "cuda" and not args.no_amp
    if args.val_frequency < 1:
        raise ValueError("--val-frequency must be >= 1")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1")
    print(f"Batch size: {args.batch_size}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Pinned memory: {pin_memory}")
    print(f"Mixed precision: {use_amp}")
    print(f"Validation frequency: every {args.val_frequency} epoch(s)")
    if args.early_stopping_patience is not None:
        print(f"Early stopping patience: {args.early_stopping_patience} validation(s)")
    checkpoint_dir, validation_results_dir = resolve_run_dirs(args)
    
    # Create datasets
    dataset = MaizeDatasetCOCO(
        Path(args.train_data_dir), 
        Path(args.train_annotation_file),
        transforms=get_transform(train=True)
    )
    
    dataset_val = MaizeDatasetCOCO(
        Path(args.val_data_dir),
        Path(args.val_annotation_file),
        transforms=get_transform(train=False)
    )

    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    # Create data loaders
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=generator,
        **loader_kwargs,
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if args.seed is not None else None,
        **loader_kwargs,
    )
    
    # Initialize model and optimizer
    model = build_model(NUM_CLASSES)
    model.to(device)
    for module in model.backbone.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    try:
        best_score = 0.0
        validations_without_improvement = 0
        best_metric_name = "AP@0.5" if args.best_metric == "ap50" else "mAP@0.5:0.95"
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch}/{args.epochs}")
            print("-" * 20)
            
            train_loss = train_one_epoch(
                model,
                optimizer,
                data_loader,
                device,
                epoch,
                scaler=scaler,
                use_amp=use_amp,
                clear_cache=args.clear_cache_each_step,
            )
            print(f"Train Loss: {train_loss:.4f}")

            should_validate = (epoch % args.val_frequency == 0) or (epoch == args.epochs - 1)
            val_stats = None
            val_score = None
            if should_validate:
                val_stats = evaluate(
                    model,
                    data_loader_val,
                    device,
                    epoch=epoch,
                    train_loss=train_loss,
                    results_dir=validation_results_dir,
                    clear_cache=args.clear_cache_each_step,
                )
                # COCOeval indexing: stats[0] is mAP@0.5:0.95, stats[1] is AP@0.5.
                # The original script selected checkpoints by stats[1]; keep that as
                # the default for manuscript consistency unless --best-metric map50_95 is used.
                val_score = val_stats[1] if args.best_metric == "ap50" else val_stats[0]
                print(f"Validation {best_metric_name}: {val_score:.4f}")
            else:
                print(f"Skipping validation this epoch (--val-frequency {args.val_frequency})")
            
            lr_scheduler.step()
            
            # Save checkpoint
            if val_score is not None and val_score > best_score:
                best_score = val_score
                validations_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric_name,
                    'best_score': best_score,
                    'best_ap50': val_stats[1],
                    'best_map50_95': val_stats[0],
                    'seed': args.seed,
                }, checkpoint_dir / 'best_model.pt')
                print(f"Saved new best model at epoch {epoch}: {best_metric_name}={best_score:.4f}")
            elif val_score is not None:
                validations_without_improvement += 1
                print(
                    f"No validation improvement for {validations_without_improvement} "
                    f"validation(s); best {best_metric_name}: {best_score:.4f}"
                )
            
            if epoch % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_score': val_score,
                    'seed': args.seed,
                }
                if val_stats is not None:
                    checkpoint.update({
                        'AP_50': val_stats[1],
                        'mAP_50_95': val_stats[0],
                    })
                torch.save(checkpoint, checkpoint_dir / f'model_epoch_{epoch}.pt')

            if (
                args.early_stopping_patience is not None
                and validations_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}: no improvement for "
                    f"{validations_without_improvement} validation(s). "
                    f"Best {best_metric_name}: {best_score:.4f}"
                )
                break
                
    except Exception as e:
        print(f"Training error: {e}")
        torch.save(model.state_dict(), checkpoint_dir / 'emergency_save.pt')

if __name__ == "__main__":
    main()
