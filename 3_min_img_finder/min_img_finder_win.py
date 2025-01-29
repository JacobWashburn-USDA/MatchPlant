"""
Script Name: min_img_finder_win
Purpose: To find the minimum set of drone images needed to cover a target area while 
         maintaining specified overlap requirements for Windon OS
Author: Worasit Sangjan
Date Created: 27 January 2025
Version: 1.0
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from shapely.geometry import box, Polygon, MultiPolygon
from dataclasses import dataclass
from datetime import datetime
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
    """Initial configuration window"""
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 8))
        maximize_window(self.fig)
        self.setup_window()
        self.error_text = None

    def setup_window(self):
        """Setup the initial configuration window"""
        self.ax = self.fig.add_subplot(111)
        self.ax.set_position([0.15, 0.2, 0.7, 0.7])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add border around the content
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
        # Main title
        self.ax.text(0.5, 1.05, 'Minimum Images Finder',
                    ha='center', va='center',
                    fontsize=20, fontweight='bold',
                    transform=self.ax.transAxes)

        self._setup_path_section()
        self._setup_parameters_section()
        self._setup_start_button()
        
        # Error message area
        self.error_ax = self.fig.add_axes([0.15, 0.02, 0.7, 0.06])
        self.error_ax.set_xticks([])
        self.error_ax.set_yticks([])
        for spine in self.error_ax.spines.values():
            spine.set_visible(False)

    def _setup_path_section(self):
        """Setup the path configuration section"""
        # Section title
        self.ax.text(0.5, 0.925, "Path Configuration",
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    transform=self.ax.transAxes)

        # Orthophoto selection
        self.ax.text(0.145, 0.85, "Orthophoto:",
                    ha='left', va='center',
                    transform=self.ax.transAxes)
        
        ortho_box = self.fig.add_axes([0.325, 0.78, 0.35, 0.03])
        self.ortho_input = TextBox(ortho_box, '', initial='')
        
        ortho_button_ax = self.fig.add_axes([0.685, 0.78, 0.08, 0.03])
        self.ortho_button = Button(ortho_button_ax, 'Browse')
        self.ortho_button.on_clicked(lambda x: self.browse_file('ortho'))

        # Orthorectified folder selection
        self.ax.text(0.03, 0.78, "Orthorectified Image Folder:",
                    ha='left', va='center',
                    transform=self.ax.transAxes)
        
        rect_box = self.fig.add_axes([0.325, 0.73, 0.35, 0.03])
        self.img_input = TextBox(rect_box, '', initial='')
        
        rect_button_ax = self.fig.add_axes([0.685, 0.73, 0.08, 0.03])
        self.img_button = Button(rect_button_ax, 'Browse')
        self.img_button.on_clicked(lambda x: self.browse_file('img'))

        # Undistorted folder selection
        self.ax.text(0.045, 0.71, "Undistorted Image Folder:",
                    ha='left', va='center',
                    transform=self.ax.transAxes)
        
        undist_box = self.fig.add_axes([0.325, 0.68, 0.35, 0.03])
        self.undist_input = TextBox(undist_box, '', initial='')
        
        undist_button_ax = self.fig.add_axes([0.685, 0.68, 0.08, 0.03])
        self.undist_button = Button(undist_button_ax, 'Browse')
        self.undist_button.on_clicked(lambda x: self.browse_file('undist'))

    def _setup_parameters_section(self):
        """Setup the optimization parameters section"""
        # Section title
        self.ax.text(0.5, 0.57, "Optimization Parameters",
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    transform=self.ax.transAxes)

        # Common settings
        input_width = 0.15
        box_height = 0.03
        label_x = 0.35
        input_x = 0.425

        self.param_inputs = {}

        # Flight Line Width
        self.ax.text(label_x, 0.48, "Flight Line Width (%):",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        input_box = self.fig.add_axes([input_x, 0.525, input_width, box_height])
        self.param_inputs["Flight Line Width (%):"] = TextBox(input_box, '', initial="15")

        # Horizontal Min Overlap
        self.ax.text(label_x, 0.41, "Horizontal Min Overlap (%):",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        input_box = self.fig.add_axes([input_x, 0.475, input_width, box_height])
        self.param_inputs["Horizontal Min Overlap (%):"] = TextBox(input_box, '', initial="1")

        # Horizontal Max Overlap
        self.ax.text(label_x, 0.34, "Horizontal Max Overlap (%):",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        input_box = self.fig.add_axes([input_x, 0.425, input_width, box_height])
        self.param_inputs["Horizontal Max Overlap (%):"] = TextBox(input_box, '', initial="15")

        # Vertical Min Overlap
        self.ax.text(label_x, 0.27, "Vertical Min Overlap (%):",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        input_box = self.fig.add_axes([input_x, 0.375, input_width, box_height])
        self.param_inputs["Vertical Min Overlap (%):"] = TextBox(input_box, '', initial="1")

        # Vertical Max Overlap
        self.ax.text(label_x, 0.20, "Vertical Max Overlap (%):",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        input_box = self.fig.add_axes([input_x, 0.325, input_width, box_height])
        self.param_inputs["Vertical Max Overlap (%):"] = TextBox(input_box, '', initial="15")

        # Uncovered Threshold
        self.ax.text(label_x, 0.13, "Uncovered Threshold (%):",
                    ha='right', va='center',
                    transform=self.ax.transAxes)
        input_box = self.fig.add_axes([input_x, 0.275, input_width, box_height])
        self.param_inputs["Uncovered Threshold (%):"] = TextBox(input_box, '', initial="10")
        
    def browse_file(self, type_):
        """Handle file and folder selection using tkinter"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        if type_ == 'ortho':
            file_path = filedialog.askopenfilename(
                title='Select Orthophoto',
                filetypes=[('TIF files', '*.tif')]
            )
            if file_path:
                self.ortho_input.set_val(file_path)
        elif type_ in ['img', 'undist']:
            folder_path = filedialog.askdirectory(
                title=f'Select {"Image" if type_ == "img" else "Undistorted Images"} Folder'
            )
            if folder_path:
                if type_ == 'img':
                    self.img_input.set_val(folder_path)
                else:
                    self.undist_input.set_val(folder_path)

        self.fig.canvas.draw_idle()

    def _setup_start_button(self):
        """Setup the start button"""
        button_ax = self.fig.add_axes([0.35, 0.1, 0.35, 0.07])
        self.button = Button(button_ax, 'Click to Start Finding',
                            color='#90EE90',
                            hovercolor='#7CCD7C')
        self.button.on_clicked(self.validate_and_start)
        self.button.label.set_fontsize(11)    

    def validate_and_start(self, event):
        """Validate all inputs and start the optimization"""
        # Validate paths
        paths = {
            'orthophoto': self.ortho_input.text.strip(),
            'image folder': self.img_input.text.strip(),
            'undistorted folder': self.undist_input.text.strip()
        }
        
        # Check paths exist
        for name, path in paths.items():
            if not path or not os.path.exists(path):
                self.show_error_message(f"Invalid or missing {name}")
                return

        # Validate and collect parameters
        config = {
            'orthophoto_path': paths['orthophoto'],
            'image_folder': paths['image folder']
        }

        for label, input_box in self.param_inputs.items():
            try:
                value = float(input_box.text.strip())
                if not 1 <= value <= 100:
                    raise ValueError
                config[label] = value
            except ValueError:
                self.show_error_message(f"Invalid {label} (must be between 1 and 100)")
                return

        optimizer = MinimumCoverageOptimizer(
            orthophoto_path=config['orthophoto_path'],
            image_folder=config['image_folder']
        )
        
        optimizer.set_parameters(
            bin_percentage=config['Flight Line Width (%):'],
            overlap_min=config['Horizontal Min Overlap (%):'],
            overlap_max=config['Horizontal Max Overlap (%):'],
            vertical_overlap_min=config['Vertical Min Overlap (%):'],
            vertical_overlap_max=config['Vertical Max Overlap (%):'],
            uncovered_threshold=config['Uncovered Threshold (%):']
        )
        
        results = optimizer.optimize_coverage() # Run optimization to get results
        plt.close(self.fig) # Close the current window
        
        # Create and show the results window in a new matplotlib figure
        result_window = ResultWindow(optimizer, results, paths['undistorted folder'])
        result_window.show()
        
    def show_error_message(self, message):
        """Display error message in the initial window"""
        if self.error_text:
            self.error_text.remove()
        
        self.error_text = self.error_ax.text(0.5, 0.5, message, color='red',
                                           ha='center', va='center',
                                           fontsize=10, wrap=True)
        self.fig.canvas.draw_idle()

        timer = self.fig.canvas.new_timer(interval=3000)
        timer.add_callback(self.clear_error_message)
        timer.start()

    def clear_error_message(self):
        """Clear the error message"""
        if self.error_text:
            self.error_text.remove()
            self.error_text = None
            self.fig.canvas.draw_idle()

    def show(self):
        """Display the window"""
        plt.show()

@dataclass
class ImageInfo:
    """Store essential image information"""
    name: str
    center_x: float
    center_y: float
    bounding_box: tuple
    bounding_polygon: Polygon
    image_width: float = None
    image_height: float = None
    distance_to_corner: float = None

class MinimumCoverageOptimizer:
    """Optimize drone image coverage with minimum overlap requirements"""
    def __init__(self, orthophoto_path: str, image_folder: str):
        """Initialize optimizer with paths and default parameters"""
        self.orthophoto_path = orthophoto_path
        self.image_folder = image_folder
        self.ortho_dataset = None
        self.ortho_bounds = None
        self.image_info_df = None
        self.reference_point = None
        self.smallest_height = None
        self.center_point_bin_percentage = None
        self.overlap_min = None
        self.overlap_max = None
        self.uncovered_area_threshold = None

    def set_parameters(self, bin_percentage, overlap_min, overlap_max, 
                    vertical_overlap_min, vertical_overlap_max,
                    uncovered_threshold):
        """Set optimization parameters"""
        self.center_point_bin_percentage = bin_percentage / 100
        self.overlap_min = overlap_min / 100  
        self.overlap_max = overlap_max / 100  
        self.vertical_overlap_min = vertical_overlap_min / 100 
        self.vertical_overlap_max = vertical_overlap_max / 100 
        self.uncovered_area_threshold = uncovered_threshold

    def _load_data(self):
        """Load and prepare image data"""
        if not os.path.exists(self.orthophoto_path):
            return False

        self.ortho_dataset = rasterio.open(self.orthophoto_path)
        self.ortho_bounds = self.ortho_dataset.bounds
        
        rectified_path_list = glob.glob(os.path.join(self.image_folder, '*.tif'))
        if not rectified_path_list:
            return False
        
        # Load image information
        image_info_list = []
        for path in rectified_path_list:
            with rasterio.open(path) as img:
                bbox = img.bounds
                polygon = box(*bbox)
                center = polygon.centroid
                image_info = {
                    'name': os.path.basename(path),
                    'center_x': center.x,
                    'center_y': center.y,
                    'bounding_box': bbox,
                    'bounding_polygon': polygon
                }
                image_info_list.append(image_info)
        
        # Create DataFrame and calculate additional metrics
        self.image_info_df = pd.DataFrame(image_info_list)
        return self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate image metrics and find reference point"""
        if self.image_info_df is None or self.image_info_df.empty:
            return False 
              
        lower_left_corner = (self.ortho_bounds.left, self.ortho_bounds.bottom)

        self.image_info_df['distance_to_corner'] = self.image_info_df['bounding_box'].apply(
            lambda bb: np.sqrt((bb.left - lower_left_corner[0])**2 + 
                             (bb.bottom - lower_left_corner[1])**2)
        )   
        self.image_info_df['image_width'] = self.image_info_df['bounding_box'].apply(
            lambda bb: bb.right - bb.left
        )
        self.image_info_df['image_height'] = self.image_info_df['bounding_box'].apply(
            lambda bb: bb.top - bb.bottom
        )
        
        # Find reference point and smallest height
        self.reference_point = self.image_info_df.loc[self.image_info_df['distance_to_corner'].idxmin()]
        self.smallest_height = self.image_info_df['image_height'].min()
        return True

    def _filter_lowest_points(self):
        """Filter images by lowest center points in bins"""
        x_width = self.reference_point['image_width'] * self.center_point_bin_percentage
        bins = np.arange(
            self.image_info_df['center_x'].min(),
            self.image_info_df['center_x'].max() + x_width,
            x_width
        )
        
        self.image_info_df['x_bin'] = pd.cut(
            self.image_info_df['center_x'],
            bins=bins,
            include_lowest=True
        )
        
        # Fix deprecation warnings
        grouped = self.image_info_df.groupby('x_bin', observed=True)
        lowest_points = grouped.apply(
            lambda df: df.loc[df['center_y'].idxmin()],
            include_groups=False
        )
        
        if self.reference_point.name not in lowest_points.index:
            lowest_points = pd.concat([lowest_points, self.reference_point.to_frame().T])
        return lowest_points.sort_values(by='center_x').reset_index(drop=True)
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap area between two bounding boxes"""
        poly1 = box(*bbox1)
        poly2 = box(*bbox2)
        intersection = poly1.intersection(poly2)
        
        return (intersection.area / poly1.area, intersection.area / poly2.area)
    
    def _calculate_vertical_overlap(self, bbox1, bbox2):
        """Calculate vertical overlap percentage between two bounding boxes"""
        min_top = min(bbox1.top, bbox2.top)
        max_bottom = max(bbox1.bottom, bbox2.bottom)
        intersection_height = min_top - max_bottom
        
        if intersection_height <= 0:
            return 0, 0
        
        # Calculate heights
        height1 = bbox1.top - bbox1.bottom
        height2 = bbox2.top - bbox2.bottom
        
        # Calculate overlap percentages relative to each image
        overlap1 = (intersection_height / height1) * 100
        overlap2 = (intersection_height / height2) * 100
        return overlap1, overlap2   
    
    def _find_sequence(self, lowest_points: pd.DataFrame, reference: pd.Series) -> list:
        """Find sequence of images with appropriate overlap"""
        current_image = reference
        sequence = [current_image]
        current_index = reference.name

        while True:
            # Get valid candidate images based on position and overlap criteria
            candidates = lowest_points[
                (lowest_points['center_x'] > current_image['center_x']) & 
                (lowest_points.index != current_index)
            ]
            
            found = False
            # Select the best candidate image based on overlap criteria
            for idx, candidate in candidates.iterrows():
                overlap1, overlap2 = self._calculate_overlap(
                    current_image['bounding_box'],
                    candidate['bounding_box']
                )
                
                if self.overlap_min <= overlap2 <= self.overlap_max:
                    sequence.append(candidate)
                    current_image = candidate
                    current_index = idx
                    found = True
                    break
                    
            if not found:
                break

        return sequence
    
    def _check_vertical_overlap(self, candidate_image: pd.Series, previous_sequence: pd.DataFrame) -> bool:
        """Check vertical overlap between candidate image and previous sequence"""
        # Get candidate image boundaries
        candidate_left = candidate_image['bounding_box'].left
        candidate_right = candidate_image['bounding_box'].right
        
        # Filter relevant images from row below
        relevant_images = previous_sequence[
            (previous_sequence['bounding_box'].apply(lambda x: x.right) > candidate_left) &
            (previous_sequence['bounding_box'].apply(lambda x: x.left) < candidate_right)
        ]
        
        if relevant_images.empty:
            print(f"Warning: No images found below candidate {candidate_image['name']}")
            return False
            
        # Check vertical overlap with relevant images
        for _, below_image in relevant_images.iterrows():
            vert_overlap1, vert_overlap2 = self._calculate_vertical_overlap(
                candidate_image['bounding_box'],
                below_image['bounding_box']
            )
            
            if not (self.vertical_overlap_min <= vert_overlap2/100 <= self.vertical_overlap_max):
                return False
                
        return True        

    def _find_first_left_image(self, previous_sequence: pd.DataFrame, all_sequences_df: pd.DataFrame) -> pd.Series:
        """Find the first (leftmost) image for the new row"""
        # Check undiscovered area
        uncovered_area = self._calculate_uncovered_area(all_sequences_df)
        uncovered_percentage = (uncovered_area.area / box(*self.ortho_bounds).area) * 100

        if uncovered_percentage <= self.uncovered_area_threshold:
            return None
            
        # Get the leftmost image from the previous sequence
        leftmost_prev = previous_sequence.iloc[0]
        # Get images that are above the previous row
        max_center_y = previous_sequence['center_y'].max()
        candidate_images = self.image_info_df[
            self.image_info_df['center_y'] > max_center_y
        ].copy()
        
        if candidate_images.empty:
            return None
        
        # Find valid candidates based ONLY on vertical overlap
        valid_candidates = []
        for _, image in candidate_images.iterrows():
            # Check vertical overlap with leftmost previous image
            vert_overlap1, vert_overlap2 = self._calculate_vertical_overlap(
                image['bounding_box'],
                leftmost_prev['bounding_box']
            )
            
            if (self.vertical_overlap_min <= vert_overlap2/100 <= self.vertical_overlap_max):
                valid_candidates.append({
                    'image': image,
                    'center_x': image['center_x'],
                    'vertical_overlap': vert_overlap2
                })
       
        if not valid_candidates:
            return None
        
        # Sort candidates by x-coordinate to find the leftmost one
        valid_candidates.sort(key=lambda x: x['center_x'])
        return valid_candidates[0]['image']
   
    def _find_sequence_above(self, leftmost_image: pd.Series, previous_sequence: pd.DataFrame, 
                             candidate_images: pd.DataFrame) -> list:
        """Find sequence of images above previous sequence."""
        current_image = leftmost_image
        sequence = [current_image]
        min_spacing = self.reference_point['image_width'] * (1 - self.overlap_max)
        
        while True:
            # Get all remaining candidates to the right
            remaining_candidates = candidate_images[
                (candidate_images['center_x'] > current_image['center_x']) &
                (~candidate_images.index.isin([img.name for img in sequence]))
            ]
            
            if remaining_candidates.empty:
                break
                
            valid_candidates = []
            # First try with strict horizontal overlap
            for _, candidate in remaining_candidates.iterrows():
                # Check vertical overlap first
                if not self._check_vertical_overlap(candidate, previous_sequence):
                    continue
                    
                # Check horizontal overlap
                horiz_overlap1, horiz_overlap2 = self._calculate_overlap(
                    current_image['bounding_box'],
                    candidate['bounding_box']
                )
                
                if self.overlap_min <= horiz_overlap2 <= self.overlap_max:
                    # Calculate score based on optimal spacing
                    ideal_x = current_image['center_x'] + min_spacing
                    position_score = abs(candidate['center_x'] - ideal_x)
                    
                    valid_candidates.append({
                        'image': candidate,
                        'score': position_score,
                        'overlap': horiz_overlap2
                    })
            
            if not valid_candidates:
                break
                
            # Choose candidate with best score
            next_image = min(valid_candidates, key=lambda x: x['score'])['image']
            
            # Check if this would be the final image
            next_candidates = candidate_images[
                (candidate_images['center_x'] > next_image['center_x']) &
                (~candidate_images.index.isin([img.name for img in sequence + [next_image]]))
            ]
            
            # If this would be the final image (no more valid candidates after)
            if next_candidates.empty:
                # Check if it overlaps too much with current sequence
                total_overlap = 0
                for seq_img in sequence:
                    _, overlap = self._calculate_overlap(
                        seq_img['bounding_box'],
                        next_image['bounding_box']
                    )
                    total_overlap += overlap
                    
                # If it is the final image and has too much overlap, stop here
                if total_overlap > self.overlap_max:
                    break
            
            sequence.append(next_image)
            current_image = next_image
        
        return sequence 
    
    def _find_images_above(self, previous_sequence: pd.DataFrame, all_sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Find images above previous sequence"""
        # Find the first left image
        leftmost_image = self._find_first_left_image(previous_sequence, all_sequences_df)
        
        if leftmost_image is None:
            return pd.DataFrame()
        
        # Get all potential candidates above the previous row
        max_center_y = previous_sequence['center_y'].max()
        candidate_images = self.image_info_df[
            self.image_info_df['center_y'] > max_center_y
        ].copy()
        
        # Find sequence using improved method
        sequence = self._find_sequence_above(leftmost_image, previous_sequence, candidate_images)
        
        if sequence:
            result_df = pd.DataFrame(sequence)
            return result_df.sort_values(by='center_x').reset_index(drop=True)
        
        return pd.DataFrame()

    def optimize_coverage(self) -> pd.DataFrame:
        """Main optimization method to find optimal image coverage"""
        # Load and prepare data
        self._load_data()
        
        # Find initial sequence
        lowest_points = self._filter_lowest_points()
        sequence = self._find_sequence(lowest_points, self.reference_point)
        sequence_df = pd.DataFrame(sequence)
        sequence_df['sequence_number'] = 1
        sequence_df['image_sequence_number'] = range(1, len(sequence_df) + 1)
        
        all_sequences_df = sequence_df.copy()
        sequence_number = 2
        
        # Calculate initial uncovered area
        uncovered_area = self._calculate_uncovered_area(all_sequences_df)
        uncovered_percentage = (uncovered_area.area / box(*self.ortho_bounds).area) * 100
        
        # Add sequences until coverage threshold is met
        while uncovered_percentage > self.uncovered_area_threshold:
            prev_sequence = all_sequences_df[
                all_sequences_df['sequence_number'] == sequence_number - 1
            ]
            
            # Pass all_sequences_df to _find_images_above
            above_sequence = self._find_images_above(prev_sequence, all_sequences_df)
            
            if above_sequence.empty:
                break
            
            new_sequence_df = above_sequence.copy()
            new_sequence_df['sequence_number'] = sequence_number
            new_sequence_df['image_sequence_number'] = range(1, len(new_sequence_df) + 1)
            
            all_sequences_df = pd.concat([all_sequences_df, new_sequence_df])
            
            uncovered_area = self._calculate_uncovered_area(all_sequences_df)
            uncovered_percentage = (uncovered_area.area / box(*self.ortho_bounds).area) * 100
            
            sequence_number += 1
        
        # Reorder columns
        cols = ['sequence_number', 'image_sequence_number'] + [
            col for col in all_sequences_df.columns 
            if col not in ['sequence_number', 'image_sequence_number']
        ]
        return all_sequences_df[cols]

    def _calculate_uncovered_area(self, sequences_df: pd.DataFrame) -> Polygon:
        """Calculate uncovered area from sequence of images"""
        ortho_polygon = box(*self.ortho_bounds)
        selected_polygons = [
            box(*row['bounding_box']) 
            for _, row in sequences_df.iterrows()
        ]
        union = selected_polygons[0]
        for poly in selected_polygons[1:]:
            union = union.union(poly)
        return ortho_polygon.difference(union)

    def save_results_and_copy_images(self, results_df: pd.DataFrame, undistorted_folder: str, save_dir: str = None, figure=None):
        """Save results to CSV and copy selected undistorted images to a new folder"""
        if save_dir is None:
            save_dir = os.path.dirname(self.image_folder)

        # Create output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)        
            
        # Clean up results DataFrame [Remove any hidden file entries]
        results_df = results_df[~results_df['name'].str.startswith('._')].copy()
            
        # Create timestamped folder/csv path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        selected_images_dir = os.path.join(save_dir, f'selected_undistorted_images_{timestamp}')
        csv_path = os.path.join(save_dir, f'selected_images_list_{timestamp}.csv')
        plot_path = os.path.join(save_dir, f'coverage_plot_{timestamp}.png')

        # Create the directory
        os.makedirs(selected_images_dir, exist_ok=True)
                
        # Get list of actual files in undistorted folder
        available_files = [f for f in os.listdir(undistorted_folder) 
                        if not f.startswith('._') and 
                        (f.endswith('.tif') or f.endswith('.TIF') or 
                        f.endswith('.jpg') or f.endswith('.JPG'))]
                
        # Create mapping of base names to actual files
        file_mapping = {}
        for f in available_files:
            base_name = os.path.splitext(os.path.splitext(f)[0])[0]  # Remove both extensions
            file_mapping[base_name.upper()] = f  # Store with uppercase for case-insensitive matching
                
        # Save relevant columns to CSV
        output_columns = [
            'sequence_number', 
            'image_sequence_number', 
            'name',
            'center_x', 
            'center_y'
        ]
        results_df[output_columns].to_csv(csv_path, index=False)
                
        # Copy undistorted images to new directory
        copied_files = 0
        for _, row in results_df.iterrows():
            # Get base name without extensions and convert to uppercase for matching
            base_name = os.path.splitext(os.path.splitext(row['name'])[0])[0].upper()
            if base_name in file_mapping:
                src_file = file_mapping[base_name]
                src = os.path.join(undistorted_folder, src_file)
                dst = os.path.join(selected_images_dir, src_file)
                shutil.copy2(src, dst)
                copied_files += 1

        # Save the plot if figure is provided
        if figure is not None:
            figure.savefig(plot_path, bbox_inches='tight', dpi=300)
        else:
            plot_path = None
                
        if copied_files > 0:
            return csv_path, selected_images_dir, plot_path
            
        return None, None, None
                    
class ResultWindow:
    """Class to handle the result window and save functionality"""
    def __init__(self, optimizer, results_df, undistorted_folder):
        self.optimizer = optimizer
        self.results_df = results_df
        self.undistorted_folder = undistorted_folder

        self.fig = plt.figure(figsize=(14, 8))
        maximize_window(self.fig)
        
        self.save_button = None
        self.save_button_ax = None
        self.message_ax = None
        
        # Calculate coverage
        self.uncovered_area = self._calculate_uncovered_area()
        self.coverage_percentage = self._calculate_coverage_percentage()       

        self.setup_window()
        self.fig.canvas.draw_idle()

    def setup_window(self):
        """Set up the result window"""
        # Create main plot axes with specific size and position
        self.ax = self.fig.add_axes([0.1, 0.25, 0.75, 0.685])

        # Setup different sections
        self._setup_background()
        self._setup_image_boundaries()
        self._setup_uncovered_area()
        self._setup_axes_labels()
        self._setup_message_area()
        self._setup_save_button()

    def _calculate_uncovered_area(self):
        """Calculate uncovered area once for reuse"""
        return self.optimizer._calculate_uncovered_area(self.results_df)

    def _calculate_coverage_percentage(self):
        """Calculate coverage percentage from uncovered area"""
        ortho_bounds = box(*self.optimizer.ortho_bounds)
        uncovered_percentage = (self.uncovered_area.area / ortho_bounds.area) * 100
        return 100 - uncovered_percentage        

    def _setup_background(self):   
        """Setup the orthophoto as background"""
        ortho_data = self.optimizer.ortho_dataset.read([1, 2, 3])
        ortho_data = np.dstack(ortho_data)
        extent = [
            self.optimizer.ortho_bounds.left, self.optimizer.ortho_bounds.right,
            self.optimizer.ortho_bounds.bottom, self.optimizer.ortho_bounds.top
        ]
        self.ax.imshow(ortho_data, extent=extent, origin='upper')
        
    def _setup_image_boundaries(self):    
        """Plot image boundaries and center points"""
        for sequence_number in self.results_df['sequence_number'].unique():
            sequence = self.results_df[self.results_df['sequence_number'] == sequence_number]
            self._plot_sequence_boundaries(sequence, sequence_number)

    def _plot_sequence_boundaries(self, sequence, sequence_number):
        """Plot boundaries for a specific sequence"""
        for idx, row in sequence.iterrows():
            # plot boundary
            x, y = row['bounding_polygon'].exterior.xy
            self.ax.plot(x, y, 'b-', lw=1, 
                label='Boundary' if sequence_number == 1 and idx == 0 else "")
            
            # Plot center point
            self.ax.plot(row['center_x'], row['center_y'], 'o', 
                color='lightcoral', markersize=5,
                label='Center of Image' if sequence_number == 1 and idx == 0 else "")
            
            # Add sequence number label
            self.ax.text(row['center_x'], row['center_y'],
                f"{row['sequence_number']}.{row['image_sequence_number']}",
                color='black', fontsize=10, ha='center', va='center')
             
    def _setup_uncovered_area(self):
        """Calculate and plot uncovered area"""
        self._plot_uncovered_area(self.uncovered_area)
    
    def _plot_uncovered_area(self, uncovered_area):
        """Plot the uncovered area on the othophoto background"""
        if isinstance(uncovered_area, MultiPolygon):
            self._plot_multipolygon(uncovered_area)
        else:
            self._plot_single_polygon(uncovered_area)

    def _plot_multipolygon(self, multipolygon):
        """Plot a multipolygon object"""
        for geom in multipolygon.geoms:
            if geom.is_valid and not geom.is_empty and geom.area > 0:
                self._plot_single_polygon(geom)
    
    def _plot_single_polygon(self, polygon):
        """Plot a single polygon object"""
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
            x, y = polygon.exterior.xy
            self.ax.plot(x, y, 'r-', lw=2, label='Uncovered Area', zorder=5)
                
            if hasattr(polygon, 'interiors'):
                for interior in polygon.interiors:
                    x, y = interior.xy
                    self.ax.plot(x, y, 'r-', lw=2, zorder=5)
    
    def _setup_axes_labels(self):
        """Set title and axes labels"""
        self.ax.set_title('Sequences of Images and Uncovered Area', 
                          fontsize=20, fontweight='bold', pad=20)
        self.ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        
    def _setup_message_area(self):
        """"Setup the message area with information"""
        self.message_ax = self.fig.add_axes([0.1, 0.1, 0.7, 0.1])
        self.message_ax.axis('off')
        
        # Display initial message text
        save_dir = os.path.dirname(self.optimizer.image_folder)
        self.show_message(
            f"""
            Results Summary:
            -----------------------------------------
            Coverage Area: {self.coverage_percentage:.2f}%
            -----------------------------------------
            Click 'Save Results' to save output files to:
            {save_dir}""")
    
    def _setup_save_button(self):
        """Setup the save button"""
        self.save_button_ax = self.fig.add_axes([0.4, 0.02, 0.15, 0.05])
        self.save_button = Button(self.save_button_ax, 'Save Results',
                                color='#90EE90', hovercolor='#7CCD7C')
        
        self.save_button.on_clicked(self.validate_and_save)
        self.save_button.label.set_fontsize(11)

    def validate_and_save(self, event):
        """Handle save button click event"""
        # Visual feedback - disable button
        self.save_button.color = '#cccccc'
        self.fig.canvas.draw_idle()

        # Create tkinter root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
                    
        # Use the optimizer's save function for all files
        save_dir = os.path.dirname(self.optimizer.image_folder)
        csv_path, images_dir, plot_path = self.optimizer.save_results_and_copy_images(
            self.results_df, 
            self.undistorted_folder,
            save_dir,
            figure=self.fig
        )
                    
        if csv_path and images_dir:
        # Update message to show success
            success_msg = f"""
            Files saved successfully:
            ------------------------------------------
            CSV: {os.path.basename(csv_path)}
            Images: {os.path.basename(images_dir)}
            Plot: {os.path.basename(plot_path)}
            """
            self.show_message(success_msg)          
        else:
            self.show_message("Error: No files were saved.")

        # Re-enable button
        self.save_button.color = '#90EE90'
        self.fig.canvas.draw_idle()

    def show_message(self, message):
        """Update message text in the window with consistent formatting"""
        if hasattr(self.message_ax, 'texts') and len(self.message_ax.texts) > 0:
            self.message_ax.texts[0].remove()
        
        self.message_ax.text(0.05, 0.7, message,
            transform=self.message_ax.transAxes,
            va='center', fontsize=10, family='monospace')
        self.fig.canvas.draw_idle()   

    def show(self):
        """Display the result window"""
        plt.show(block=True)

def main():
    """Main program execution"""
    initial_window = InitialWindow()
    initial_window.show()

if __name__ == "__main__":
    main()