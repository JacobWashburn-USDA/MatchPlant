"""
Script Name: img_splitter_win
Purpose: To create the designed tiles from original image size for Window OS
Author: Worasit Sangjan
Date Created: 31 January 2025
Version: 1.0
"""

import os
import cv2
import json
import math
from pathlib import Path
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from matplotlib.widgets import Button, TextBox
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog

def maximize_window(fig):
    """Windows-specific window maximization"""
    backend = plt.get_backend()
    if backend in ['TkAgg', 'Qt5Agg']:
        fig.canvas.manager.window.state('zoomed')
    else:
        fig.canvas.manager.window.showMaximized()

class InitialWindow:
    """Initial configuration window for image annotation settings"""
    def __init__(self):
        # Create figure size
        self.fig = plt.figure(figsize=(14, 8))
        maximize_window(self.fig)
        
        # Create main axes' position and size
        self.ax = self.fig.add_subplot(111)
        # Set the axes of the figure space
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add border around the content
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
        # Create error message axes position
        self.error_ax = self.fig.add_axes([0.1, 0.02, 0.8, 0.06])
        self.error_ax.set_xticks([])
        self.error_ax.set_yticks([])
        for spine in self.error_ax.spines.values():
            spine.set_visible(False)

        self.ready_to_process = False
        self.error_text = None
        self.setup_window()

    def setup_window(self):
        """Setup the initial configuration window"""
        # Main title 
        self.ax.text(0.5, 1.05, 'Image Annotation Configuration',
                     ha='center', va='center',
                     fontsize=20, fontweight='bold',
                     transform=self.ax.transAxes)

        # Setup all sections
        self._setup_path_section()
        self._setup_split_section()
        self._setup_tile_section()
        self._setup_start_button()

    def _setup_path_section(self):
        """Setup the path configuration section"""
        # Section title
        self.ax.text(0.5, 0.95, "Path Configuration", 
                     ha='center', va='center', 
                     fontsize=14, fontweight='bold',
                     transform=self.ax.transAxes)
        
        # Image folder
        self.ax.text(0.225, 0.875, "Image Folder:", 
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        
        img_box = self.fig.add_axes([0.3, 0.78, 0.4, 0.035])
        self.img_input = TextBox(img_box, '', initial='')
        
        img_button_ax = self.fig.add_axes([0.72, 0.78, 0.1, 0.035])
        self.img_button = Button(img_button_ax, 'Browse')
        self.img_button.on_clicked(lambda x: self.browse_file('img'))

        # Annotation folder
        self.ax.text(0.225, 0.795, "Annotation Folder:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        
        ann_box = self.fig.add_axes([0.3, 0.715, 0.4, 0.035])
        self.ann_input = TextBox(ann_box, '', initial='')
        
        ann_button_ax = self.fig.add_axes([0.72, 0.715, 0.1, 0.035])
        self.ann_button = Button(ann_button_ax, 'Browse')
        self.ann_button.on_clicked(lambda x: self.browse_file('ann'))

        # Output folder
        self.ax.text(0.225, 0.705, "Output Folder:", 
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        
        out_box = self.fig.add_axes([0.3, 0.645, 0.4, 0.035])
        self.out_input = TextBox(out_box, '', initial='')
        
        out_button_ax = self.fig.add_axes([0.72, 0.645, 0.1, 0.035])
        self.out_button = Button(out_button_ax, 'Browse')
        self.out_button.on_clicked(lambda x: self.browse_file('out'))

    def _setup_split_section(self):
        """Setup the dataset split configuration section"""
        # Section title
        self.ax.text(0.5, 0.615, "Dataset Split Configuration",
                     ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     transform=self.ax.transAxes)
        
        # Train split
        self.ax.text(0.225, 0.545, "Train Split:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        train_box = self.fig.add_axes([0.3, 0.515, 0.2, 0.035])
        self.train_split = TextBox(train_box, '', initial='0.7')

        # Validation split
        self.ax.text(0.225, 0.47, "Validation Split:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        val_box = self.fig.add_axes([0.3, 0.455, 0.2, 0.035])
        self.val_split = TextBox(val_box, '', initial='0.15')

        # Test split
        self.ax.text(0.225, 0.395, "Test Split:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        test_box = self.fig.add_axes([0.3, 0.395, 0.2, 0.035])
        self.test_split = TextBox(test_box, '', initial='0.15')

    def _setup_tile_section(self):
        """Setup the tile configuration section"""
        # Section title
        self.ax.text(0.5, 0.305, "Tile Configuration", 
                     ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     transform=self.ax.transAxes)

        # Left side controls
        # Target size
        self.ax.text(0.225, 0.237, "Target Tile Size:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        target_box = self.fig.add_axes([0.3, 0.27, 0.2, 0.035])
        self.target_size = TextBox(target_box, '', initial='1024')

        # Overlap pixels
        self.ax.text(0.225, 0.16, "Overlap Pixels:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        overlap_box = self.fig.add_axes([0.3, 0.205, 0.2, 0.035])
        self.overlap = TextBox(overlap_box, '', initial='100')

        # Right side controls
        # Minimum IoU
        self.ax.text(0.66, 0.237, "Minimum IoU:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        iou_box = self.fig.add_axes([0.65, 0.27, 0.2, 0.035])
        self.min_iou = TextBox(iou_box, '', initial='0.3')

        # Minimum size
        self.ax.text(0.66, 0.16, "Minimum Size:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        size_box = self.fig.add_axes([0.65, 0.205, 0.2, 0.035])
        self.min_size = TextBox(size_box, '', initial='10')

    def browse_file(self, type_):
        "Handle image, annotation, and output selection"
        root = tk.Tk()
        root.withdraw()

        if type_ == 'img':
            folder_path = filedialog.askdirectory(
                title='Select Image Folder'
            )
            if folder_path:
                self.img_input.set_val(folder_path)
        elif type_ in ['ann', 'out']:
            folder_path = filedialog.askdirectory(
                title=f'Select {"Annotation" if type_ == "ann" else "Output"} Folder'
            )
            if folder_path:
                if type_ == 'ann':
                    self.ann_input.set_val(folder_path)
                else:
                    self.out_input.set_val(folder_path)
        
        self.fig.canvas.draw_idle()
    
    def _setup_start_button(self):
        """Setup the start button"""
        button_ax = self.fig.add_axes([0.325, 0.115, 0.35, 0.06])
        self.button = Button(button_ax, 'Click to Start Processing',
                           color='#90EE90',
                           hovercolor='#7CCD7C')
        self.button.on_clicked(self.validate_and_start)
        self.button.label.set_fontsize(12)           

    def validate_and_start(self, event):
        paths = {
            'img_dir': self.img_input.text.strip(),
            'ann_dir': self.ann_input.text.strip(),
            'out_dir': self.out_input.text.strip()
        }
        
        if not all(paths.values()):
            self.show_error_message("All paths must be specified")
            return
        
        if not all(os.path.exists(path) for path in paths.values()):
            self.show_error_message("All paths must exist")
            return
        
        splits = {
            'train': float(self.train_split.text),
            'val': float(self.val_split.text),
            'test': float(self.test_split.text)
        }
        
        if not 0.99 < sum(splits.values()) < 1.01:
            self.show_error_message("Split ratios must sum to 1.0")
            return
        
        if not all(0 < split < 1 for split in splits.values()):
            self.show_error_message("Split ratios must be between 0 and 1")
            return
        
        tile_config = {
            'target_size': int(self.target_size.text),
            'overlap_pixels': int(self.overlap.text),
            'min_iou': float(self.min_iou.text),
            'min_size': int(self.min_size.text)
        }
        
        if tile_config['target_size'] < 100:
            self.show_error_message("Target tile size must be at least 100 pixels")
            return
        
        self.config = {
            'paths': paths,
            'split_config': splits,
            'tile_config': tile_config
        }
        
        self.ready_to_process = True

    def show_error_message(self, message):
        """Display error message in the window"""
        if self.error_text:
            self.error_text.remove()
        
        self.error_text = self.error_ax.text(0.5, 0.5, message,
                                             color='red',
                                             ha='center', va='center',
                                             fontsize=14, wrap=True)
        self.fig.canvas.draw_idle()

    def get_config(self):
        plt.show(block=False)
        while plt.fignum_exists(self.fig.number) and not self.ready_to_process:
            plt.pause(0.1)
        return getattr(self, 'config', None)

def initialize_config():
    window = InitialWindow()
    return window.get_config()

@dataclass
class SplitConfig:
    train: float
    val: float
    test: float
    
    def validate(self) -> bool:
        return abs(sum((self.train, self.val, self.test)) - 1.0) < 1e-6

class TileConfig:
    """Configuration for image tiling."""
    target_size: int = 1024       # Target tile size
    overlap_pixels: int = 100     # Overlap between tiles
    min_iou: float = 0.3          # Minimum overlap required
    min_size: int = 10            # Minimum bbox dimension

    def calculate_grid(self, image_width: int, image_height: int) -> Tuple[int, int]:
        """Calculate optimal grid size based on image dimensions."""
        # Handle small images
        if image_width <= self.target_size and image_height <= self.target_size:
            return 1, 1
            
        # Calculate grid size
        grid_x = max(1, math.ceil(image_width / self.target_size))
        grid_y = max(1, math.ceil(image_height / self.target_size))
        return grid_x, grid_y
    
    def calculate_tile_size(self, image_width: int, image_height: int) -> Tuple[int, int]:
        """Calculate tile size based on grid."""
        grid_x, grid_y = self.calculate_grid(image_width, image_height)
        tile_w = image_width // grid_x
        tile_h = image_height // grid_y
        return tile_w, tile_h

def validate_and_adjust_bbox(bbox: List[float], 
                           quad_dims: Tuple[int, int, int, int],
                           tile_config: TileConfig) -> Optional[List[float]]:
    """Original robust bbox adjustment logic."""
    x, y, w, h = bbox
    qx_min, qy_min, qx_max, qy_max = quad_dims
    
    # Calculate box coordinates 
    box_x1, box_y1 = x, y
    box_x2, box_y2 = x + w, y + h
    
    # Calculate intersection
    inter_x1 = max(box_x1, qx_min) 
    inter_y1 = max(box_y1, qy_min)
    inter_x2 = min(box_x2, qx_max)
    inter_y2 = min(box_y2, qy_max)
    
    # Calculate intersection area
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_area = inter_w * inter_h
    box_area = w * h
    
    # Calculate IoU
    iou = inter_area / box_area
    
    # Convert to tile coordinates
    new_x = inter_x1 - qx_min
    new_y = inter_y1 - qy_min
    new_w = inter_x2 - inter_x1
    new_h = inter_y2 - inter_y1
    
    # Use thresholds from tile_config
    if new_w < tile_config.min_size or new_h < tile_config.min_size or iou < tile_config.min_iou:
        return None
        
    return [new_x, new_y, new_w, new_h]

def split_image_and_adjust_coco(
    image_path: Path,
    annotation_path: Path,
    output_dir: Path,
    split: str,
    start_id: int,
    tile_config: TileConfig
) -> Optional[Dict]:
    """Split image into tiles with adaptive grid sizing."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
        
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    h, w, _ = image.shape
        
    # Calculate optimal grid and tile sizes
    grid_x, grid_y = tile_config.calculate_grid(w, h)
    tile_w, tile_h = tile_config.calculate_tile_size(w, h)

    # Create tiles with overlap
    tiles = {}
    for i in range(grid_y):
        for j in range(grid_x):
            # Calculate tile boundaries with overlap
            x_min = max(0, j * tile_w - tile_config.overlap_pixels)
            y_min = max(0, i * tile_h - tile_config.overlap_pixels)
            x_max = min(w, (j + 1) * tile_w + tile_config.overlap_pixels)
            y_max = min(h, (i + 1) * tile_h + tile_config.overlap_pixels)
                
            # Store tile coordinates
            tiles[f"r{i+1}c{j+1}"] = (x_min, y_min, x_max, y_max)

    # Create output directory
    data_dir = output_dir / "data" / split
    data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize COCO format output
    combined_coco = {
        "categories": coco_data.get("categories", [{
            "id": 1,
            "name": "object",
            "supercategory": "object"
        }]),
        "images": [],
        "annotations": []
    }

    # Process each tile
    ann_id_counter = 1
    for idx, (tile_name, tile_dims) in enumerate(tiles.items(), 0):
        image_id = start_id + idx
        qx_min, qy_min, qx_max, qy_max = tile_dims
            
        # Extract tile
        tile = image[qy_min:qy_max, qx_min:qx_max]
        tile_filename = f"{image_path.stem}_{tile_name}.tif"
            
        # Add tile to images list
        combined_coco["images"].append({
            "id": image_id,
            "file_name": tile_filename,
            "width": qx_max - qx_min,
            "height": qy_max - qy_min,
            "date_captured": "",
            "tile_info": {
                "original_width": w,
                "original_height": h,
                "tile_position": tile_name,
                "grid_size": f"{grid_x}x{grid_y}"
            }
        })

        # Process annotations for this tile
        for ann in coco_data["annotations"]:
            adjusted_bbox = validate_and_adjust_bbox(ann["bbox"], tile_dims, tile_config)
            if adjusted_bbox:
                x, y, w, h = adjusted_bbox
                combined_coco["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": image_id,
                    "category_id": ann.get("category_id", 1),
                    "bbox": adjusted_bbox,
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id_counter += 1

        # Save tile
        cv2.imwrite(str(data_dir / tile_filename), tile)
            
    return combined_coco
        
def process_dataset(
    original_img_dir: Path,
    annotation_dir: Path,
    output_base_dir: Path,
    split_config: SplitConfig,
    tile_config: TileConfig = TileConfig(),
    seed: int = 42
) -> Dict:
    """Process dataset and return statistics for results window."""
    if not split_config.validate():
        raise ValueError("Split ratios must sum to 1.0")
    
    np.random.seed(seed)
    results = {}  # Store results for display

    # Find all JSON annotation files
    json_files = list(annotation_dir.glob("*_annotations.json"))
    image_pairs = []
    
    # Collect valid image-annotation pairs
    for json_file in json_files:
        img_base = json_file.stem.replace('_annotations', '')
        # Search for the image file 
        for img_file in original_img_dir.glob(f"{img_base}*"):
            if img_file.suffix.lower() in ['.tif', '.jpg', '.png']:
                image_pairs.append((img_file, json_file))
                break
    
    if not image_pairs:
        raise ValueError("No valid image-annotation pairs found")
    
    # Split into train/val/test
    np.random.shuffle(image_pairs)
    n_total = len(image_pairs)
    n_train = int(n_total * split_config.train)
    n_val = int(n_total * split_config.val)
    
    splits = {
        'train': image_pairs[:n_train],
        'val': image_pairs[n_train:n_train + n_val],
        'test': image_pairs[n_train + n_val:]
    }
    
    # Create output directory
    annotation_dir = output_base_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split with reset IDs
    for split_name, pairs in splits.items():
        if not pairs:
            continue
            
        all_annotations = []
        # Reset counters for each split
        image_id_counter = 0  # Start images from 0
        
        # Process each image in the split
        for image_path, json_file in pairs:
            # Get image dimensions for logging
            annotations = split_image_and_adjust_coco(
                image_path,
                json_file,
                output_base_dir,
                split_name,
                image_id_counter,
                tile_config
            )
            
            if annotations:
                all_annotations.append(annotations)
                img = cv2.imread(str(image_path))
                if img is not None:
                    h, w = img.shape[:2]
                    grid_x, grid_y = tile_config.calculate_grid(w, h)
                    image_id_counter += grid_x * grid_y
        
        # Combine annotations with reset annotation IDs
        combined = {
            "images": [],
            "categories": all_annotations[0]["categories"],
            "annotations": []
        }
        
        # Reset annotation ID counter for each split
        annotation_id_counter = 1  # COCO annotations typically start from 1
        
        # Combine while maintaining split-specific IDs
        for ann in all_annotations:
            combined["images"].extend(ann["images"])
            for annotation in ann["annotations"]:
                annotation["id"] = annotation_id_counter
                annotation_id_counter += 1
                combined["annotations"].append(annotation)
        
        # Save annotations
        output_path = annotation_dir / f"{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2)
            
        # Store statistics for results window
        image_ids = [img["id"] for img in combined["images"]]
        annotation_ids = [ann["id"] for ann in combined["annotations"]]
        
        results[split_name] = {
            'images': {
                'count': len(combined["images"]),
                'id_range': (min(image_ids), max(image_ids))
            },
            'annotations': {
                'count': len(combined["annotations"]),
                'id_range': (min(annotation_ids), max(annotation_ids))
            }
        }
        
    return results

class ResultsWindow:
    """Window to display processing results and statistics"""
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 8))
        maximize_window(self.fig)
        self.setup_window()

    def setup_window(self):
        """Setup the results window"""
        self.ax = self.fig.add_subplot(111)
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add border
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
        # Title
        self.ax.text(0.5, 0.925, 'Processing Results',
                    ha='center', va='center',
                    fontsize=20, fontweight='bold',
                    transform=self.ax.transAxes)

        # Create close button
        button_ax = self.fig.add_axes([0.35, 0.025, 0.3, 0.05])
        self.close_button = Button(button_ax, 'Close',
                                 color='#90EE90',
                                 hovercolor='#7CCD7C')
        self.close_button.on_clicked(self._close_window)
        self.close_button.label.set_fontsize(12)  

    def display_results(self, results):
        """Display processing results"""
        y_pos = 0.85  # Starting position for text
        
        for split_name, stats in results.items():
            # Split name
            self.ax.text(0.1, y_pos, f"{split_name} dataset information:",
                        ha='left', va='center',
                        fontsize=12, fontweight='bold',
                        transform=self.ax.transAxes)
            y_pos -= 0.05
            
            # Images statistics
            img_stats = stats['images']
            self.ax.text(0.15, y_pos,
                        f"Images: {img_stats['count']} "
                        f"(IDs: {img_stats['id_range'][0]} to {img_stats['id_range'][1]})",
                        ha='left', va='center',
                        fontsize=11,
                        transform=self.ax.transAxes)
            y_pos -= 0.05
            
            # Annotations statistics
            ann_stats = stats['annotations']
            self.ax.text(0.15, y_pos,
                        f"Annotations: {ann_stats['count']} "
                        f"(IDs: {ann_stats['id_range'][0]} to {ann_stats['id_range'][1]})",
                        ha='left', va='center',
                        fontsize=11,
                        transform=self.ax.transAxes)
            y_pos -= 0.1  # Extra space between splits

        self.fig.canvas.draw_idle()

    def _close_window(self, event):
        """Handle close button click"""
        plt.close(self.fig)

    def show(self):
        """Display the window"""
        plt.show()

if __name__ == "__main__":
    # Get configuration from initial window
    initial_window = InitialWindow()
    config = initial_window.get_config()
    
    if config is None:
        plt.close(initial_window.fig)
        exit(1)
    
    # Create path objects
    original_img_dir = Path(config['paths']['img_dir'])
    annotation_dir = Path(config['paths']['ann_dir'])
    output_base_dir = Path(config['paths']['out_dir'])
    
    # Create split config
    split_config = SplitConfig(
        train=config['split_config']['train'],
        val=config['split_config']['val'],
        test=config['split_config']['test']
    )
    
    # Create tile config
    tile_config = TileConfig()
    tile_config.target_size = config['tile_config']['target_size']
    tile_config.overlap_pixels = config['tile_config']['overlap_pixels']
    tile_config.min_iou = config['tile_config']['min_iou']
    tile_config.min_size = config['tile_config']['min_size']
    
    # Process dataset
    results = process_dataset(
        original_img_dir=original_img_dir,
        annotation_dir=annotation_dir,
        output_base_dir=output_base_dir,
        split_config=split_config,
        tile_config=tile_config
    )
        
    # Show results window
    results_window = ResultsWindow()
    results_window.display_results(results)
    
    # Close initial window only after showing results
    plt.close(initial_window.fig)
    
    # Show results window
    results_window.show()
        