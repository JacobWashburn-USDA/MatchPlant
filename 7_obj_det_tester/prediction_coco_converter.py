from pathlib import Path
import json
import torch
from typing import Dict, List, Any

class COCOPredictionConverter:
    """Converts model predictions to COCO format annotations."""
    
    def __init__(self, test_annotation_path: str, output_dir: Path):
        self.output_dir = Path(output_dir)
        
        # Load test annotations to maintain structure
        with open(test_annotation_path, 'r') as f:
            self.test_coco = json.load(f)
            
        # Create image id to info mapping for quick lookup
        self.image_info = {img['id']: img for img in self.test_coco['images']}
        
        # Initialize prediction collection
        self.predictions = {
            "images": self.test_coco['images'],
            "categories": self.test_coco['categories'],
            "annotations": []
        }
        self.annotation_id = 1

    def process_predictions(self, predictions: List[Dict], 
                          targets: List[Dict],
                          confidence_threshold: float = 0.5) -> None:
        """Process batch of predictions into COCO format."""
        for pred, target in zip(predictions, targets):
            image_id = target['image_id'].item()
            
            # Process boxes above confidence threshold
            pred_boxes = pred['boxes'].cpu()
            pred_scores = pred['scores'].cpu()
            pred_labels = pred['labels'].cpu()  # Predicted labels
            
            # Add predictions to COCO format
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score < confidence_threshold:
                    continue
                    
                x1, y1, x2, y2 = box.tolist()
                width = x2 - x1
                height = y2 - y1
                
                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": label.item(), 
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "score": score.item(),
                    "iscrowd": 0,
                    "segmentation": []
                }
                
                self.predictions['annotations'].append(annotation)
                self.annotation_id += 1

    def save_annotations(self, output_name: str = 'predictions.json') -> None:
        """Save collected predictions in COCO format."""
        output_path = self.output_dir / output_name
        self.output_dir.mkdir(exist_ok=True, parents=True)  # Ensure directory exists
        with open(output_path, 'w') as f:
            json.dump(self.predictions, f, indent=2)

def create_prediction_annotations(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    test_annotation_path: str,
    output_dir: Path,
    confidence_threshold: float = 0.5,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> None:
    """Create COCO format annotations from model predictions."""
    converter = COCOPredictionConverter(test_annotation_path, output_dir)
    
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            predictions = model(images)
            
            converter.process_predictions(
                predictions,
                targets,
                confidence_threshold
            )
    
    converter.save_annotations()
    print(f"Saved predictions to {output_dir}/predictions.json")