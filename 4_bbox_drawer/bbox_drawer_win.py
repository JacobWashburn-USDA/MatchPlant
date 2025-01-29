"""
Script Name: bbox_drawer_win
Purpose: A tool for drawing bounding boxes on images with multi-category support, 
         designed for image annotation tasks with COCO/YOLO format output.
         Implemented for Windows OS.
Authors: Worasit Sangjan
Date Created: 28 January 2025
Version: 1.0
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
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
    """Initial configuration window for path selection"""
    def __init__(self):
        plt.ion() # Enable interactive mode
        # Setup window
        self.fig = plt.figure(figsize=(14, 8))      
        maximize_window(self.fig)          

        # Initialize state
        self.error_text = None
        self.image_folder = None
        self.num_categories = 1  # Default to single category
        self.category_inputs = {}
        self.category_labels = {} 
        self.selected_format = "COCO" # Default format

        self.setup_window()
        # Initialize error section
        self._setup_error_section()

    def setup_window(self):
        """Setup the initial configuration window"""
        self.ax = self.fig.add_subplot(111)
        self.ax.set_position([0.15, 0.2, 0.7, 0.7])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add border
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
        # Main title
        self.ax.text(0.5, 1.065, 'Boundary Box Drawing Tool',
                    ha='center', va='center',
                    fontsize=20, fontweight='bold',
                    transform=self.ax.transAxes)

        # Setup components
        self._setup_path_section()
        self._setup_format_section() 
        self._setup_category_section()
        self._setup_start_button()

    def _setup_path_section(self):
        """Setup the path configuration section"""
        # Header
        self.ax.text(0.5, 0.93, "Image Directory Selection",
                    ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    transform=self.ax.transAxes)
        
        # Description
        self.ax.text(0.5, 0.875, 
                    """Select the folder containing your selected images
                    *Output folders will be created at the same location  of selected image folder""",
                    ha='center', va='center',
                    fontsize=10, style='italic',
                    transform=self.ax.transAxes)
        
        # Image directory
        self.ax.text(0.24, 0.80, "Image Folder:",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        
        # Directory input and button
        dir_box = self.fig.add_axes([0.325, 0.745, 0.35, 0.03])
        self.dir_input = TextBox(dir_box, '', initial='')
        
        dir_button_ax = self.fig.add_axes([0.685, 0.745, 0.08, 0.03])
        self.dir_button = Button(dir_button_ax, 'Browse')
        self.dir_button.on_clicked(self.browse_directory)

    def browse_directory(self, event):
        """Handle directory selection"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        folder_path = filedialog.askdirectory(title='Select Image Folder')
        if folder_path:
            self.image_folder = os.path.abspath(folder_path)
            self.dir_input.set_val(self.image_folder)

    def _setup_format_section(self):
        """Setup format selection"""
        # Format selection label
        self.ax.text(0.24, 0.72, "Output Format:",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        
        # Create radio buttons for format selection
        format_ax = self.fig.add_axes([0.325, 0.665, 0.1, 0.06])
        self.format_radio = RadioButtons(
            format_ax, 
            ('COCO', 'YOLO'),
            active=0
        )
        self.format_radio.on_clicked(self.update_format)

    def update_format(self, label):
        """Update selected format"""
        self.selected_format = label

    def _setup_category_section(self):
        """Setup section for selecting number of categories using figure coordinates"""
        # Label 
        self.ax.text(0.24, 0.615, "Number of Categories (1-5):",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        
        # Position input box and update button
        cat_count_box = self.fig.add_axes([0.325, 0.615, 0.1, 0.03])
        self.cat_count_input = TextBox(cat_count_box, '', initial='1')
        self.cat_count_input.on_submit(self.update_category_count)

        update_button_ax = self.fig.add_axes([0.435, 0.615, 0.08, 0.03])
        self.update_button = Button(update_button_ax, 'Update')
        self.update_button.on_clicked(lambda x: self.update_category_count(self.cat_count_input.text))

        # Setup initial category inputs
        self._setup_category_inputs()

    def _setup_category_inputs(self):
        """Setup category input fields using figure coordinates"""
        # Clear existing inputs
        for ax in list(self.category_inputs.values()):
            ax.ax.remove()
        self.category_inputs.clear()

        # Clear existing labels
        for label in self.category_labels.values():
            label.remove()
        self.category_labels.clear()

        # Define consistent spacing and positioning
        start_y = 0.58
        vertical_gap = 0.07
        box_x, box_width, box_height = 0.325, 0.35, 0.03

        # Create category inputs
        for i in range(self.num_categories):
            y_pos = start_y - (i * vertical_gap)
            
            # Label 
            label = self.ax.text(0.24, y_pos-0.03, f"Category {i+1}:", 
                                 ha='right', va='center', 
                                 transform=self.ax.transAxes)  # Use figure transform
            self.category_labels[i] = label 

            # Text box
            cat_box = self.fig.add_axes([box_x, start_y - (i * 0.05) - 0.01, box_width, box_height])
            # Set initial value and style
            initial_value = 'maize' if i == 0 and self.num_categories == 1 else f'plant_{i+1}'
            self.category_inputs[i] = TextBox(cat_box, '', initial=initial_value)
            
            # Style text box
            self.category_inputs[i].ax.set_facecolor('white')
            for spine in self.category_inputs[i].ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)

        self.fig.canvas.draw_idle() # Force a redraw to update the display

    def update_category_count(self, text):
        """Update number of category input fields"""
        try:
            num = int(text)
            if 1 <= num <= 5:
                self.num_categories = num
                self._setup_category_inputs()  # Refresh category inputs
                self.fig.canvas.draw_idle()
            else:
                self.show_error_message("Number of categories must be between 1 and 5")
                self.cat_count_input.set_val('1')
        except ValueError:
            self.show_error_message("Please enter a valid number")
            self.cat_count_input.set_val('1')    

    def _setup_start_button(self):
        """Setup the start button"""
        button_ax = self.fig.add_axes([0.35, 0.24, 0.3, 0.07])
        self.button = Button(button_ax, 'Click to Start Drawing',
                           color='#90EE90',
                           hovercolor='#7CCD7C')
        self.button.on_clicked(self.validate_and_start)
        self.button.label.set_fontsize(11)

    def validate_and_start(self, event):
        """Validate inputs and start the application"""
        image_folder = self.dir_input.text.strip()
        if not image_folder:
            return self.show_error_message("Please select an image folder")
            
        self.image_folder = image_folder
        # Get categories from inputs
        categories = [input_box.text.strip() for input_box in self.category_inputs.values()]
        
        # Check for empty categories
        if any(not cat for cat in categories):
            return  self.show_error_message("All category names must be filled")
            
        # Check for duplicate categories
        if len(set(categories)) != len(categories):
            return self.show_error_message("Category names must be unique")
            
        # Check if directory contains images
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        if not any(f.lower().endswith(image_extensions) for f in os.listdir(self.image_folder)):
            return self.show_error_message("No supported image files found")

        plt.close(self.fig)

    def _setup_error_section(self):
        """Error message area"""
        self.error_ax = self.fig.add_axes([0.15, 0.02, 0.7, 0.06])
        self.error_ax.set_xticks([])
        self.error_ax.set_yticks([])
        for spine in self.error_ax.spines.values():
            spine.set_visible(False)

    def show_error_message(self, message):
        """Display error message in the window"""
        if self.error_text:
            self.error_text.remove()
        
        self.error_text = self.error_ax.text(0.5, 0.5, message, color='red',
                                           ha='center', va='center',
                                           fontsize=10, wrap=True)
        self.fig.canvas.draw_idle()

    def show(self):
        """Display the window and return selected path and categories"""
        # Force draw and show the window
        self.fig.canvas.draw()
        plt.show(block=True)  # Use block=True to keep window open
        categories = [input_box.text.strip() for input_box in self.category_inputs.values()]
        return self.image_folder, categories, self.selected_format

class InteractiveBoxProcessor:
    def __init__(self, image_folder: str, category_name: str, categories: list, selected_format: str):
        """Initialize the interactive box processing system"""
        self.image_folder = Path(image_folder)
        parent_dir = self.image_folder.parent
        self.selected_format = selected_format   
        
        # Set up output directories at the same level as image folder
        self.box_masks_dir = parent_dir / "mask_box"
        self.img_box_dir = parent_dir / "img_box" 
        os.makedirs(self.box_masks_dir, exist_ok=True)
        os.makedirs(self.img_box_dir, exist_ok=True)

        # Add category cycling
        self.categories = categories
        self.current_category_index = 0
        self.current_category = categories[0] # Default to first category
        self.category_to_id = {cat: idx + 1 for idx, cat in enumerate(categories)} # Category id mapping

        # Initialize COCO format data structure
        self.coco_format = {
            "images": [],
            "categories": [
                {
                "id": i + 1,
                "name": cat,  
                "supercategory": "plant"
                }
                for i, cat in enumerate(categories)
            ],
            "annotations": []
        }
        
        # COCO format counters (1-based indexing)
        self.image_id = 1
        self.annotation_id = 1

        # Drawing state
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.box_count = 0
        self.first_time_center_text = True
        self.emergency_stop = False
        self.mouse_x, self.mouse_y = 0, 0
        self.current_image = None
        self.current_image_name = None
        self.hide_center_text = False
        self.show_instructions = True
        self.drawn_boxes = []  # List to store (x1, y1, x2, y2) tuples

    def get_category_color(self, category_id):
        """Get unique color for each category"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255)   # Magenta
        ]
        return colors[(category_id - 1) % len(colors)]

    def draw_box_with_label(self, img, box_info, box_number):
        """Draw a box with its label number and category"""
        x1, y1, x2, y2, category = box_info
        
        # Get color for this category
        color = self.get_category_color(self.category_to_id[category])
        
        # Draw the rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
        
        # Prepare label text
        label = f"#{box_number} {category}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Calculate label background position
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        
        # Draw white background for text
        cv2.rectangle(img, 
                     (label_x, label_y - text_height - 5),
                     (label_x + text_width + 10, label_y + 5),
                     (255, 255, 255),
                     -1)
        
        # Draw text
        cv2.putText(img, label,
                    (label_x + 5, label_y),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for box drawing"""
        self.mouse_x, self.mouse_y = x, y

        # Use right-click for drawing
        if event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.first_time_center_text = False
            return 

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:  # Update while drawing with right button
                img_copy = self.current_image.copy()
                for i, box in enumerate(self.drawn_boxes):
                    self.draw_box_with_label(img_copy, box, i+1)
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 0, 255), 5)
                self.overlay_text(img_copy, True)
                self.simulate_pointer(img_copy, x, y)
                cv2.imshow('image', img_copy)

        elif event == cv2.EVENT_RBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, x2 = min(self.ix, x), max(self.ix, x)
                y1, y2 = min(self.iy, y), max(self.iy, y)
                self.drawn_boxes.append((x1, y1, x2, y2, self.current_category))
                self.process_single_box((x1, y1, x2, y2))
                self.box_count += 1

    def process_single_box(self, bbox):
        """Process a single box and save mask"""
        if self.current_image is None or self.current_image_name is None:
            return

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Create base name and files
        base_name = Path(self.current_image_name).stem
        box_number = self.box_count + 1
        mask_name = f"{base_name}_{box_number}.JPG"
        
        # Save mask image always
        height, width = self.current_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        masked_img = cv2.bitwise_and(self.current_image, self.current_image, mask=mask)
        mask_path = self.box_masks_dir / mask_name
        cv2.imwrite(str(mask_path), masked_img)

        if self.selected_format == "YOLO":
            # Convert to YOLO format and save individual annotation
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            class_id = self.categories.index(self.current_category)
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            
            txt_name = f"{base_name}_{box_number}.txt"
            txt_path = self.box_masks_dir / txt_name
            with open(txt_path, 'w') as f:
                f.write(yolo_line)

            # Save classes.txt if not exists
            classes_file = self.box_masks_dir / "classes.txt"
            if not classes_file.exists():
                with open(classes_file, 'w') as f:
                    f.write('\n'.join(self.categories))
        else:
            # Original COCO format annotation
            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": self.category_to_id[self.current_category],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []
            }
            
            self.coco_format["annotations"].append(annotation)
            self.annotation_id += 1
            
            # Save individual box annotation
            json_name = f"{base_name}_{box_number}.json"
            box_annotation = {
                "images": [{
                    "id": self.image_id,
                    "file_name": self.current_image_name,
                    "width": width,
                    "height": height,
                    "date_captured": ""
                }],
                "categories": self.coco_format["categories"],
                "annotations": [annotation]
            }
            
            json_path = self.box_masks_dir / json_name
            with open(json_path, 'w') as f:
                json.dump(box_annotation, f, indent=2)

    def overlay_text(self, image, drawing_mode=False, hide_center_text=False):
        """Add instruction text overlay"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        text_color = (0, 0, 0)
        thickness = 4
        glow_color = (255, 255, 255)
        glow_thickness = 15

        # Calculate positions for both texts in the same row
        y_offset = 75  # Starting y position
        
        # Draw toggle instruction on the left
        toggle_instruction = "Press 'T' to toggle instructions on/off |"
        (toggle_width, toggle_height), _ = cv2.getTextSize(toggle_instruction, font, font_scale, thickness)
        
        # Draw toggle instruction with padding from left
        toggle_x = 20
        self.draw_glowing_text(image, toggle_instruction, (toggle_x, y_offset), font, 
                            font_scale, thickness, text_color, glow_color, glow_thickness)
        
        # Draw category text after toggle instruction with some padding
        category_text = f"Current Category: {self.current_category_index+1}-{self.current_category} (Press numbers 1-{len(self.categories)} to switch)"
        padding = 50  # Space between the two texts
        category_x = toggle_x + toggle_width + padding
        
        # Add background rectangle for category text
        (cat_width, cat_height), _ = cv2.getTextSize(category_text, font, font_scale, thickness)
        cv2.rectangle(image, 
                    (category_x - 5, y_offset - cat_height - 5),
                    (category_x + cat_width + 10, y_offset + 5),
                    self.get_category_color(self.category_to_id[self.current_category]),
                    -1)
        
        # Draw category text
        self.draw_glowing_text(image, category_text, (category_x, y_offset), font, 
                            font_scale, thickness, text_color, glow_color, glow_thickness)
        
        if not self.show_instructions:  # Skip other instructions if hidden
            return

        if drawing_mode:
            instructions = [
                "+ Draw: Right Click and drag to draw boxes",
                "+ Zoom: Use a mouse wheel to zoom in/out and Left click to move",
                "+ Delete: Press 'D' to delete the last box and its files",
                "+ Next: Press 'Enter' when finished with current image",
                "+ Exit: Press 'ESC' to stop completely",
                f"Current image: {self.current_image_name}"
            ]

            # Start other instructions below the header row
            y_offset += 60
            for text in instructions:
                self.draw_glowing_text(image, text, (20, y_offset), font, 
                                    font_scale, thickness, text_color, glow_color, glow_thickness)
                y_offset += 60

        if self.first_time_center_text:
            text1 = "Draw boundary boxes to cover plants"
            text2 = "[Instructions are provided at the right corner]"
            (text1_width, text1_height), _ = cv2.getTextSize(text1, font, 3.5, thickness)
            (text2_width, text2_height), _ = cv2.getTextSize(text2, font, 3.5, thickness)
            text1_x = (image.shape[1] - text1_width) // 2
            text2_x = (image.shape[1] - text2_width) // 2
            text1_y = (image.shape[0] - (text1_height + text2_height + 50)) // 2 + text1_height
            text2_y = text1_y + text2_height + 50
            self.draw_glowing_text(image, text1, (text1_x, text1_y), font, 3.5, 
                                thickness, text_color, glow_color, glow_thickness)
            self.draw_glowing_text(image, text2, (text2_x, text2_y), font, 3.5, 
                                thickness, text_color, glow_color, glow_thickness)

    def draw_glowing_text(self, image, text, position, font, font_scale, thickness, text_color, glow_color, glow_thickness):
        """Draw text with glow effect"""
        x, y = position
        for i in range(1, glow_thickness + 1):
            cv2.putText(image, text, (x, y), font, font_scale, glow_color, thickness + i, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def simulate_pointer(self, image, x, y):
        """Draw pointer at mouse position"""
        pointer_size = 20
        color = (0, 0, 255)
        cv2.circle(image, (x, y), pointer_size, color, -1)
        
    def save_image_with_boxes(self):
        """Save the current image with all drawn boxes"""
        if self.current_image is None or not self.drawn_boxes:
            return
            
        # Create a copy of the current image
        img_with_boxes = self.current_image.copy()
        
        # Draw all boxes with their labels
        for i, box in enumerate(self.drawn_boxes):
            self.draw_box_with_label(img_with_boxes, box, i+1)
            
        # Save the image with boxes
        output_path = self.img_box_dir / self.current_image_name
        cv2.imwrite(str(output_path), img_with_boxes)

    def delete_last_box(self):
        """Delete the last created box files and decrement counter"""
        if self.box_count > 0:
            base_name = Path(self.current_image_name).stem
            
            # Files to delete (use current box count as the index)
            mask_name = f"{base_name}_{self.box_count}.JPG"
            
            # Extension based on format
            ann_ext = ".txt" if self.selected_format == "YOLO" else ".json"
            ann_name = f"{base_name}_{self.box_count}{ann_ext}"
            
            # Delete mask file
            mask_path = self.box_masks_dir / mask_name
            if mask_path.exists():
                mask_path.unlink()
            
            # Delete annotation file
            ann_path = self.box_masks_dir / ann_name
            if ann_path.exists():
                ann_path.unlink()
            
            # Remove the last box from drawn_boxes list
            if self.drawn_boxes:
                self.drawn_boxes.pop()
            
            # Remove from COCO format if using COCO
            if self.selected_format == "COCO":
                self.annotation_id -= 1
                self.coco_format["annotations"].pop()
            
            # Decrement counter
            self.box_count -= 1
        else:
            print("No boxes to delete")

    def process_images(self):
        """Process all images in the source directory"""
        image_files = [f for f in os.listdir(self.image_folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
                    and not f.startswith('._')]
        
        if not image_files:
            print("No image files found in the selected directory")
            return
        
        for image_name in sorted(image_files):
            self.first_time_center_text = True
            
            # Load image
            image_path = str(self.image_folder / image_name)
            
            # Use IMREAD_COLOR for faster loading
            self.current_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            self.current_image_name = image_name
            self.box_count = 0
            self.drawn_boxes = []
            
            if self.current_image is None:
                continue

            # Add image info to COCO format
            height, width = self.current_image.shape[:2]
            image_info = {
                "id": self.image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
                "date_captured": ""
            }
            self.coco_format["images"].append(image_info)
            
            # Setup window
            cv2.namedWindow('image', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)  # Add GUI_NORMAL flag
            cv2.setMouseCallback('image', self.mouse_callback)
            cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
            # Get screen dimensions
            cv2.imshow('image', self.current_image)
            cv2.waitKey(500)  # Short delay to ensure window is created
            
            while True:
                if not self.drawing:
                    img_copy = self.current_image.copy()
                    for i, box in enumerate(self.drawn_boxes):
                        self.draw_box_with_label(img_copy, box, i+1)
                    self.overlay_text(img_copy, True)
                    self.simulate_pointer(img_copy, self.mouse_x, self.mouse_y)
                    cv2.imshow('image', img_copy)
                
                key = cv2.waitKey(1) & 0xFF

                # Number keys 1-5 for category selection
                if ord('1') <= key <= ord(str(len(self.categories))):
                    self.current_category_index = key - ord('1')
                    self.current_category = self.categories[self.current_category_index]
                
                elif key == ord('d') or key == ord('D'):
                    self.delete_last_box()
                elif key == ord('t') or key == ord('T'):
                    self.show_instructions = not self.show_instructions
                elif key == 13:  # Enter
                    if self.drawn_boxes:
                        self.save_image_with_boxes()
                        self.save_image_annotations()
                    break
                elif key == 27:  # ESC
                    if self.drawn_boxes:
                        self.save_image_with_boxes()
                        self.save_image_annotations()
                    self.emergency_stop = True
                    break
                
                if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                    if self.drawn_boxes:
                        self.save_image_with_boxes()
                        self.save_image_annotations()
                    self.emergency_stop = True
                    break

            if self.emergency_stop:
                print("Emergency stop triggered. Exiting...")
                break

        cv2.destroyAllWindows()

    def save_image_annotations(self):
        """Save annotations for current image in selected format"""
        if not self.current_image_name or not self.drawn_boxes:
            return

        base_name = Path(self.current_image_name).stem
        height, width = self.current_image.shape[:2]
        
        if self.selected_format == "YOLO":
            # Convert boxes to YOLO format and save
            yolo_lines = []
            for box in self.drawn_boxes:
                x1, y1, x2, y2, category = box
                # Convert to YOLO format (normalized center coordinates)
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                # Class ID is 0-based in YOLO
                class_id = self.categories.index(category)
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                yolo_lines.append(yolo_line)
                
            # Save to .txt file
            txt_path = self.img_box_dir / f"{base_name}_annotations.txt"
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
                
            # Save classes.txt if not exists
            classes_file = self.img_box_dir / "classes.txt"
            if not classes_file.exists():
                with open(classes_file, 'w') as f:
                    f.write('\n'.join(self.categories))
        
        else:  # COCO format
            # Original COCO format saving
            current_annotations = [
                ann for ann in self.coco_format["annotations"]
                if ann["image_id"] == self.image_id
            ]
            
            image_annotation = {
                "images": [next(img for img in self.coco_format["images"] 
                            if img["id"] == self.image_id)],
                "categories": self.coco_format["categories"],
                "annotations": current_annotations
            }
            
            json_path = self.img_box_dir / f"{base_name}_annotations.json"
            with open(json_path, 'w') as f:
                json.dump(image_annotation, f, indent=2) 

def main():
    # Start with the initial window
    initial_window = InitialWindow()
    image_folder, categories, selected_format = initial_window.show()
        
    if image_folder and categories:
        # Initialize and run the box processor with all categories
        processor = InteractiveBoxProcessor(
            image_folder=image_folder,
            category_name=categories[0],  # First category as default
            categories=categories,  # Pass all categories
            selected_format=selected_format
        )
        processor.process_images()

if __name__ == "__main__":
    main()
