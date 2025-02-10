"""
Script Name: gcp_finder_win
Purpose: To find drone images containing a ground control point (GCP) for Window OS
Author: Worasit Sangjan
Date Created: 2 February 2025
Version: 1.1
"""

import os
import cv2
import glob
import math
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ExifTags
from PIL import Image, ImageFile
from math import radians, sin, cos, sqrt, atan2, tan, atan
from matplotlib.pyplot import figure, show
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import RadioButtons, Button, TextBox
from matplotlib import pyplot as plt
from pyproj import Transformer
from itertools import groupby
import tkinter as tk
from tkinter import filedialog

def maximize_window(fig):
    """Maximize window for Windows"""
    backend = plt.get_backend()
    if backend in ['TkAgg', 'Qt5Agg']:
        fig.canvas.manager.window.state('zoomed')
    else:
        fig.canvas.manager.window.showMaximized()

# Global Configuration
DEFAULT_CONFIG = {
    'gcp_path': '',
    'img_folder': '',
    'distance_threshold': 10  # meters
}

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
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add border around the content
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
         # Main title
        self.ax.text(0.5, 1.05, 'GCP Finder', 
                     ha='center', va='center',
                     fontsize=20, fontweight='bold',
                     transform=self.ax.transAxes)

        self._setup_path_section()
        self._setup_camera_section()
        self._setup_threshold_section()
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
                     fontsize=14, fontweight='bold',
                     transform=self.ax.transAxes)

        # Image folder
        self.ax.text(0.265, 0.85, "Image Folder:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        
        img_box = self.fig.add_axes([0.325, 0.765, 0.35, 0.03])
        self.img_input = TextBox(img_box, '', initial='')
        
        img_button_ax = self.fig.add_axes([0.685, 0.765, 0.08, 0.03])
        self.img_button = Button(img_button_ax, 'Browse')
        self.img_button.on_clicked(lambda x: self.browse_file('img'))

        # GCP file
        self.ax.text(0.265, 0.78, "GCP File:",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        
        gcp_box = self.fig.add_axes([0.325, 0.71, 0.35, 0.03])
        self.gcp_input = TextBox(gcp_box, '', initial='')
        
        gcp_button_ax = self.fig.add_axes([0.685, 0.71, 0.08, 0.03])
        self.gcp_button = Button(gcp_button_ax, 'Browse')
        self.gcp_button.on_clicked(lambda x: self.browse_file('gcp'))

    def _setup_camera_section(self):
        """Setup the camera configuration section"""
        # Section title
        self.ax.text(0.5, 0.63, "Camera Sensor Configuration",
                     ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     transform=self.ax.transAxes)
        
        # Explanatory text
        self.ax.text(0.5, 0.59, "Enter the camera sensor dimensions (in millimeters)",
                     ha='center', va='center',
                     fontsize=10, style='italic',
                     transform=self.ax.transAxes)
        
        # Width input
        self.ax.text(0.265, 0.515, "Width (mm):",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        width_box = self.fig.add_axes([0.325, 0.5, 0.15, 0.03])
        self.width_input = TextBox(width_box, '', initial='')

        # Height input
        self.ax.text(0.595, 0.515, "Height (mm):",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        height_box = self.fig.add_axes([0.59, 0.5, 0.15, 0.03])
        self.height_input = TextBox(height_box, '', initial='')

    def _setup_threshold_section(self):
        """Setup the threshold configuration section with fixed positioning"""
        # Section title
        self.ax.text(0.5, 0.365, "Distance Threshold Configuration",
                     ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     transform=self.ax.transAxes)

        # Explanatory text
        self.ax.text(0.5, 0.325,
                     "Set the maximum distance (in meters) for searching images near GCP points",
                     ha='center', va='center',
                     fontsize=10, style='italic',
                     transform=self.ax.transAxes)
        
        # Threshold input
        self.ax.text(0.265, 0.25, "Threshold (m):",
                     ha='right', va='center',
                     transform=self.ax.transAxes)
        threshold_box = self.fig.add_axes([0.325, 0.285, 0.15, 0.03])
        self.threshold_input = TextBox(threshold_box, '', initial='10')

    def browse_file(self, type_):
        """Handle image and GCP folder selection"""
        root = tk.Tk()
        root.withdraw()

        if type_ == 'img':
            folder_path = filedialog.askdirectory(
                title='Select Image Folder'
            )
            if folder_path:
                self.img_input.set_val(folder_path)
        else:
            file_path = filedialog.askopenfilename(
                title='Select GCP File',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
            )
            if file_path:
                self.gcp_input.set_val(file_path)

    def _setup_start_button(self):
        """Setup the start button"""
        button_ax = self.fig.add_axes([0.325, 0.125, 0.35, 0.07])
        self.button = Button(button_ax, 'Click to Start Finding',
                             color='#90EE90',
                             hovercolor='#7CCD7C')
        self.button.on_clicked(self.validate_and_start)
        self.button.label.set_fontsize(12)            

    def validate_and_start(self, event):
        """Validate all inputs and start the application"""
        # Validate paths
        paths = {
            'image folder': self.img_input.text.strip(),
            'GCP file': self.gcp_input.text.strip()
        }
        
        # Check paths exist
        for name, path in paths.items():
            if not path:
                self.show_error_message(f"Please select the {name}")
                return
            if not os.path.exists(path):
                self.show_error_message(f"{name} does not exist")
                return
                
        # Check GCP file format
        if not os.path.splitext(paths['GCP file'])[1].lower() == '.csv':
            self.show_error_message("GCP file must be a CSV file")
            return

        # Validate sensor dimensions
        dimensions = {
            'width': self.width_input.text.strip(),
            'height': self.height_input.text.strip()
        }
        
        if not all(dimensions.values()):
            self.show_error_message("Please enter both width and height values")
            return
            
        sensor_values = {}
        try:
            for name, value in dimensions.items():
                val = float(value)
                if not 1 < val <= 100:
                    self.show_error_message(f"Sensor {name} must be between 1 and 100 mm")
                    return
                sensor_values[name] = val
        except ValueError:
            self.show_error_message("Invalid sensor dimensions")
            return

        # Validate threshold
        try:
            threshold_val = float(self.threshold_input.text.strip() or '5')
            if not 0 < threshold_val <= 50:
                self.show_error_message("Threshold must be between 0 and 50 meters")
                return
        except ValueError:
            self.show_error_message("Invalid threshold value")
            return

        # Create configuration and start application
        config = {
            'gcp_path': paths['GCP file'],
            'img_folder': paths['image folder'],
            'distance_threshold': threshold_val
        }

        plt.close(self.fig) # Close the current window
        image_extractor = ImageExtractor(config)  
        image_extractor.sensor_configs = {
            'DEFAULT': {
                'width': sensor_values['width'],
                'height': sensor_values['height']
            }
        }
        image_extractor.start() # Run imageextractor class

    def show_error_message(self, message):
        """Display error message in the main window"""
        if self.error_text:
            self.error_text.remove()
        
        self.error_text = self.error_ax.text(0.5, 0.5, message, color='red',
                                             ha='center', va='center',
                                             fontsize=14, wrap=True)
        self.fig.canvas.draw_idle()

    def show(self):
        """Display the window and cleanup resources"""
        plt.show()
        plt.close('all')

class CoordinateTransformer:
    """A utility class for transforming coordinates between WGS84 and UTM system"""
    def __init__(self):
        self.transformer = None
        self.utm_zone = None
        self.hemisphere = None

    def initialize_from_gcp(self, gcp_df):
        """Initial coordinate transformer"""
        if gcp_df.empty or "X" not in gcp_df.columns or "Y" not in gcp_df.columns:
            print("Invalid GCP dataframe format")
            return False
            
        sample_longitude = gcp_df["X"].iloc[0]
        sample_latitude = gcp_df["Y"].iloc[0]
        
        self.utm_zone = int((sample_longitude + 180) // 6) + 1
        self.hemisphere = "N" if sample_latitude >= 0 else "S"
        utm_crs = f"EPSG:326{self.utm_zone:02d}" if self.hemisphere == "N" else f"EPSG:327{self.utm_zone:02d}"
        
        self.transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        return True

    def transform_coordinates(self, lat, lon):
        """Convert coordinates"""
        if self.transformer is None:
            return None
        
        return self.transformer.transform(lon, lat)

class GPSUtils:
    """Universal utility class for GPS and image metadata operations"""
    EXIF_TAGS = {
        'GPS': 34853,
        'FOCAL_LENGTH': 37386,
        'XMP': 700,
        'MAKE': 271,
        'MODEL': 272,
        'RELATIVE_ALTITUDE': 'RelativeAltitude',
        'ABSOLUTE_ALTITUDE': 'AbsoluteAltitude',
        'FLIGHT_YAW_DEGREE': 'FlightYawDegree',
        'GIMBAL_PITCH_DEGREE': 'GimbalPitchDegree'
    }

    @staticmethod
    def _get_manufacturer_info(exif_data):
        """Get camera manufacturer and model"""
        if not exif_data:
            return None, None
            
        make = exif_data.get(GPSUtils.EXIF_TAGS['MAKE'], '').strip()
        model = exif_data.get(GPSUtils.EXIF_TAGS['MODEL'], '').strip()
        return make, model

    @staticmethod
    def _convert_to_degrees(value):
        """Convert GPS coordinates to degrees"""
        if not isinstance(value, tuple) or len(value) != 3:
            return None

        degrees = []
        for component in value:
            if hasattr(component, 'numerator') and hasattr(component, 'denominator'):
                degrees.append(float(component.numerator) / float(component.denominator))
            elif isinstance(component, tuple) and len(component) == 2:
                degrees.append(float(component[0]) / float(component[1]))
            elif isinstance(component, (int, float)):
                degrees.append(float(component))
            else:
                return None

        return degrees[0] + (degrees[1] / 60.0) + (degrees[2] / 3600.0)

    @staticmethod
    def _safe_rational_to_float(value):
        """Convert rational numbers to float"""
        if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
            if value.denominator == 0:
                return None
            return float(value.numerator) / float(value.denominator)
            
        if isinstance(value, tuple) and len(value) == 2:
            if value[1] == 0:
                return None
            return float(value[0]) / float(value[1])
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
                
        return None

    @staticmethod
    def _extract_xmp_metadata(xmp_data):
        """Extract metadata from XMP format"""
        if not isinstance(xmp_data, str):
            return {}

        metadata = {}
        import re
        patterns = {
            'relative_altitude': r'RelativeAltitude="([^"]+)"',
            'absolute_altitude': r'AbsoluteAltitude="([^"]+)"',
            'flight_yaw': r'FlightYawDegree="([^"]+)"',
            'gimbal_pitch': r'GimbalPitchDegree="([^"]+)"'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, xmp_data)
            if match:
                try:
                    metadata[key] = float(match.group(1))
                except ValueError:
                    continue
                    
        return metadata

    @staticmethod
    def extract_image_metadata(img_path):
        """Extract metadata from any drone image"""
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return None
            
        metadata = {
            'coordinates': None,
            'coordinate_type': None,
            'altitude': None,
            'focal_length': None,
            'dimensions': None,
            'manufacturer': None,
            'model': None,
            'roll': None,
            'pitch': None,
            'yaw': None,
            'timestamp': None
        }

        # Try loading with PIL first
        try:
            with PIL.Image.open(img_path) as img:
                metadata['dimensions'] = img.size
                exif_data = img._getexif()
                
                if exif_data:
                    metadata['manufacturer'], metadata['model'] = GPSUtils._get_manufacturer_info(exif_data)
                    
                    focal_length = exif_data.get(GPSUtils.EXIF_TAGS['FOCAL_LENGTH'])
                    if focal_length:
                        metadata['focal_length'] = GPSUtils._safe_rational_to_float(focal_length)

                    gps_info = exif_data.get(GPSUtils.EXIF_TAGS['GPS'])
                    if gps_info:
                        lat = GPSUtils._convert_to_degrees(gps_info.get(2))
                        lon = GPSUtils._convert_to_degrees(gps_info.get(4))
                        
                        if lat is not None and lon is not None:
                            lat_ref = gps_info.get(1, 'N')
                            lon_ref = gps_info.get(3, 'E')
                            lat = -lat if lat_ref == 'S' else lat
                            lon = -lon if lon_ref == 'W' else lon
                            
                            metadata.update({
                                'coordinates': (lat, lon),
                                'coordinate_type': 'wgs84'
                            })

                            alt = GPSUtils._safe_rational_to_float(gps_info.get(6))
                            if alt is not None:
                                metadata['altitude'] = -alt if gps_info.get(5, 0) == 1 else alt

                    # Get XMP data
                    xmp_data = exif_data.get(GPSUtils.EXIF_TAGS['XMP'])
                    if xmp_data:
                        xmp_metadata = GPSUtils._extract_xmp_metadata(xmp_data)
                        if metadata['altitude'] is None:
                            metadata['altitude'] = (xmp_metadata.get('absolute_altitude') or 
                                                xmp_metadata.get('relative_altitude'))
                        if metadata['yaw'] is None:
                            metadata['yaw'] = xmp_metadata.get('flight_yaw')
                        if metadata['pitch'] is None:
                            metadata['pitch'] = xmp_metadata.get('gimbal_pitch')

        except Exception as pil_error:
            # Fallback to OpenCV if PIL fails
            print(f"PIL loading failed: {pil_error}, trying OpenCV")
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 2:  # grayscale
                    metadata['dimensions'] = img.shape
                elif len(img.shape) == 3:  # BGR or BGRA
                    metadata['dimensions'] = img.shape[:2]

        return metadata

class ImageFinder:
    """Class for finding relevant images with universal dataset support"""
    def __init__(self, base_threshold=3, image_extractor=None):
        self.base_threshold = base_threshold
        self.ie = image_extractor

    def find_images_for_gcp(self, img_folder, gcp_coords, gcp_elevation):
        """Find images that likely contain the GCP using enhanced criteria"""
        if not os.path.exists(img_folder):
            print("Image folder does not exist")
            return []
            
        if not gcp_coords or len(gcp_coords) != 2:
            print("Invalid GCP coordinates")
            return []
            
        patterns = ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.TIF', '*.tif', '*.TIFF', '*.tiff']
        img_files = []
        candidates = []
        
        # Get GCP UTM coordinates for distance calculations
        gcp_utm_x, gcp_utm_y = self.ie.coord_transformer.transform_coordinates(
            gcp_coords[0], gcp_coords[1]
        )
        if gcp_utm_x is None or gcp_utm_y is None:
            print("Could not transform GCP coordinates to UTM")
            return []

        # Get all image files
        for pattern in patterns:
            pattern_path = os.path.join(img_folder, pattern)
            img_files.extend(glob.glob(pattern_path))
            # Remove duplicates while preserving order
            img_files = list(dict.fromkeys(img_files))
        
        if not img_files:
            print("No images found in folder")
            return []

        # Process each image
        for img_path in img_files:
            metadata = GPSUtils.extract_image_metadata(img_path)
            if not metadata or not metadata['coordinates']:
                continue
                
            footprint = self.calculate_image_footprint(metadata, gcp_elevation)
            if not footprint:
                continue
                
            visibility_score = self.estimate_gcp_visibility(metadata, gcp_coords, footprint)
            if visibility_score > 0:
                candidates.append({
                    'path': img_path,
                    'metadata': metadata,
                    'footprint': footprint,
                    'visibility_score': visibility_score,
                    'working_distance': footprint['working_distance']
                })

        # Sort by score and distance: Higher score first and Closer distance preferred
        return sorted(candidates, key=lambda x: (-x['visibility_score'], x['working_distance'])) 

    def calculate_gcp_position_in_image(self, metadata, gcp_coords):
        """Calculate the likely position of GCP in the image frame for any dataset type"""
        if not metadata or not metadata['coordinates']:
            return None

        # Get image coordinates in UTM
        if metadata['coordinate_type'] == 'wgs84':
            img_lat, img_lon = metadata['coordinates']
            img_utm = self.ie.coord_transformer.transform_coordinates(img_lat, img_lon)
            if not img_utm:
                return None
            img_x, img_y = img_utm
        else:
            img_x, img_y = metadata['coordinates']

        # Get GCP coordinates in UTM
        gcp_utm_x, gcp_utm_y = self.ie.coord_transformer.transform_coordinates(
            gcp_coords[0], gcp_coords[1]
        )
        if not (gcp_utm_x and gcp_utm_y):
            return None

        # Calculate position vector
        dx = gcp_utm_x - img_x
        dy = gcp_utm_y - img_y

        # Apply rotation if metadata available
        yaw = metadata.get('yaw')
        if yaw is not None:
            heading_rad = math.radians(yaw)
            x_rot = dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
            y_rot = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)
            dx, dy = x_rot, y_rot

        return dx, dy

    def calculate_image_footprint(self, metadata, gcp_elevation):
        """Calculate footprint with universal dataset support"""
        # Verify required fields
        required_fields = ['altitude', 'focal_length', 'dimensions']
        if not all(metadata.get(field) for field in required_fields):
            return None
                
        # Calculate working distance
        working_distance = metadata['altitude'] - gcp_elevation
        if working_distance <= 0:
            return None
        
        # Apply pitch correction if available
        pitch = metadata.get('pitch')
        if pitch is not None:
            pitch_rad = math.radians(pitch)
            working_distance = working_distance / math.cos(pitch_rad)
                
        # Get sensor configuration
        sensor_size = self.ie.sensor_configs.get('DEFAULT')
        if not sensor_size:
            return None
            
        # Calculate FOV and ground coverage
        horizontal_fov = 2 * math.atan(sensor_size['width'] / (2 * metadata['focal_length']))
        vertical_fov = 2 * math.atan(sensor_size['height'] / (2 * metadata['focal_length']))
        
        ground_width = 2 * working_distance * math.tan(horizontal_fov / 2)
        ground_height = 2 * working_distance * math.tan(vertical_fov / 2)
        
        return {
            'width': ground_width,
            'height': ground_height,
            'working_distance': working_distance,
            'horizontal_fov': horizontal_fov,
            'vertical_fov': vertical_fov
            }

    def estimate_gcp_visibility(self, metadata, gcp_coords, footprint):
        """Estimate visibility with advanced scoring"""
        if not footprint:
            return 0.0
        
        # Get GCP position relative to image
        gcp_pos = self.calculate_gcp_position_in_image(metadata, gcp_coords)
        if not gcp_pos:
            return 0.0
        
        x, y = gcp_pos
        # Calculate relative position
        x_rel = x / (footprint['width'] / 2)
        y_rel = y / (footprint['height'] / 2)
        
        distance_from_center = math.sqrt(x_rel**2 + y_rel**2)
        
        # Check if GCP is within image bounds
        if distance_from_center > 0.95:
            return 0.0
        
        # Calculate base score
        base_score = 1 - distance_from_center**2
        # Apply scaling factors
        height_factor = 1.0
        if metadata['altitude'] is not None:
            if metadata['altitude'] > 100:
                height_factor = 0.8
            elif metadata['altitude'] > 50:
                height_factor = 0.9

        pitch_factor = 1.0
        pitch = metadata.get('pitch')
        if pitch is not None:
            pitch_angle = abs(pitch)
            if pitch_angle > 30:
                pitch_factor = 0.7
            elif pitch_angle > 15:
                pitch_factor = 0.85

        return base_score * height_factor * pitch_factor  

class EventManager:
    """Manages all event handling for the application"""
    def __init__(self, image_extractor):
        self.ie = image_extractor
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup all event handlers"""
        # Disconnect default handlers
        self.ie.fig.canvas.mpl_disconnect(
            self.ie.fig.canvas.manager.key_press_handler_id)
        
        # Connect all handlers
        events = {
            'button_press_event': self.on_press,
            'button_release_event': self.on_release,
            'motion_notify_event': self.on_motion,
            'key_press_event': self.on_key,
            'scroll_event': self.on_scroll
        }
        
        for event, handler in events.items():
            self.ie.fig.canvas.mpl_connect(event, handler)
    
    def on_press(self, event):
        """Handle mouse press events"""
        if event.inaxes != self.ie.ax:
            return
        
        self.ie.cur_xlim = self.ie.ax.get_xlim()
        self.ie.cur_ylim = self.ie.ax.get_ylim()
        self.ie.press = (self.ie.x0, self.ie.y0, event.xdata, event.ydata)
        self.ie.x0, self.ie.y0, self.ie.xpress, self.ie.ypress = self.ie.press
        
        if event.button is MouseButton.RIGHT:
            self._handle_point_creation(event)
    
    def _handle_point_creation(self, event):
        """Handle creation of new point"""        
        utm_x, utm_y = self.ie.coord_transformer.transform_coordinates(
            self.ie.lat, self.ie.long)
        
        if utm_x is not None and utm_y is not None:
            point_data = {
                'long': self.ie.long,
                'lat': self.ie.lat,
                'elev': self.ie.elevation,
                'x': event.xdata,
                'y': event.ydata,
                'fname': self.ie.img_name,
                'id': self.ie.gcp_id,
                'utm_x': utm_x,
                'utm_y': utm_y
            }
            
            self.ie.collected_points.append(point_data)
            self.ie.clicked_list.append([
                int(event.xdata),
                int(event.ydata),
                self.ie.img_name,
                self.ie.gcp_id
            ])
            
            self.ie.coord_display.update(self.ie.clicked_list)
    
    def on_release(self, event):
        """Handle mouse release"""
        self.ie.press = None
        self.ie.ax.figure.canvas.draw()
    
    def on_motion(self, event):
        """Handle mouse motion (pan)"""
        if self.ie.press is None or event.inaxes != self.ie.ax:
            return
        
        dx = event.xdata - self.ie.xpress
        dy = event.ydata - self.ie.ypress
        self.ie.cur_xlim -= dx
        self.ie.cur_ylim -= dy
        self.ie.ax.set_xlim(self.ie.cur_xlim)
        self.ie.ax.set_ylim(self.ie.cur_ylim)
        self.ie.ax.figure.canvas.draw()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 's':
            self.ie.save_collected_points()
            print("Points saved to gcp_list.txt")
        elif event.key == 'd':
            if self.ie.clicked_list and self.ie.collected_points:
                self.ie.clicked_list.pop()
                self.ie.collected_points.pop()
                self.ie.coord_display.update(self.ie.clicked_list)
    
    def on_scroll(self, event):
        """Handle scroll events for zooming"""
        if event.inaxes != self.ie.ax:
            return
        
        cur_xlim = self.ie.ax.get_xlim()
        cur_ylim = self.ie.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        
        base_scale = 2.0
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
        
        self.ie.ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        self.ie.ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        self.ie.ax.figure.canvas.draw()

class CoordDisplay:
    """Class for displaying coordinate information"""
    def __init__(self, fig, position=[0.84, 0.11, 0.15, 0.77]):
        self.ax = fig.add_axes(position)
        self._clear_axes()
    
    def _clear_axes(self):
        """Clear and reset axes properties"""   
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
    
    def update(self, points, max_points=25):
        """Update display with points"""
        self._clear_axes()
        
        if not points:
            return
        
        visible_points = points[-max_points:][::-1] # Get last N points, reversed
        total_points = len(points)
        self.ax.text(0.05, 0.95, 'GCP Location:', fontsize=15, fontweight='bold')
        self.ax.text(0.05, 0.93, f'(showing {len(visible_points)} of {total_points})', 
                    fontsize=10, style='italic')
        
        for i, point in enumerate(visible_points): # Display points
            y_pos = 0.90 - (i * 0.04)
            text = f'[{point[0]}, {point[1]}], {point[2]}, {point[3]}'
            self.ax.text(0.05, y_pos, text, fontsize=8, verticalalignment='top')
        
        self.ax.figure.canvas.draw_idle()

class ImageExtractor:
    """"Main class for GCP controller and control UI"""
    def __init__(self, config=None):
        """Initialize the ImageExtractor"""
        self.config = config or DEFAULT_CONFIG
        self._setup_paths()
        self.reset_state()
        
        # Load GCP data
        self.gcp_df = pd.read_csv(self.gcp_path)
        self.coord_transformer = CoordinateTransformer()
        if not self.coord_transformer.initialize_from_gcp(self.gcp_df):
            print("Failed to initialize coordinate transformer")
            return
        
        # Initialize image finder
        self.image_finder = ImageFinder(
            base_threshold=self.distance_threshold,
            image_extractor=self
        )
    
    def _setup_paths(self):
        """Setup and normalize paths"""
        self.img_folder = os.path.normpath(self.config['img_folder'])
        if not self.img_folder.endswith(os.sep):
            self.img_folder += os.sep
        self.gcp_path = os.path.normpath(self.config['gcp_path'])
        self.distance_threshold = self.config['distance_threshold']
    
    def reset_state(self):
        """Reset all state variables to initial values"""
        self.press = None
        self.cur_xlim = self.cur_ylim = None
        self.x0 = self.y0 = self.x1 = self.y1 = self.xpress = self.ypress = None
        self.img_file_path = None
        self.current_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        self.img_name = None
        self.lat = self.long = None
        self.clicked_list = []
        self.collected_points = []

    def load_and_display_image(self, img_name):
        """Load and display an image with memory management"""
        if not img_name:
            return False
            
        img_path = os.path.join(self.img_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return False
            
        if hasattr(self, 'current_image'):
            del self.current_image
        
        # Load image with PIL or fallback to OpenCV
        try:
            with Image.open(img_path) as pil_img:
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                self.current_image = np.array(pil_img)
        except:
            img = cv2.imread(img_path)
            if img is None:
                print("Failed to load image")
                return False
            self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get metadata
        self.img_name = img_name
        metadata = GPSUtils.extract_image_metadata(img_path)
        if metadata and metadata['coordinates']:
            self.lat, self.long = metadata['coordinates']
        
        # Update display
        self.ax.clear()
        self.ax.imshow(self.current_image)
        height, width = self.current_image.shape[:2]
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        plt.gcf().canvas.flush_events()
        self.fig.canvas.draw_idle()
        return True

    def get_gcp_coordinates(self, gcp_id):
        """"Retrive the coordinates and elevation for a GCP"""
        if not isinstance(gcp_id, (int, str)):
            print("Invalid GCP ID")
            return None
            
        try:
            gcp_id = int(gcp_id)
        except ValueError:
            print("Invalid GCP ID format")
            return None
            
        filtered_df = self.gcp_df[self.gcp_df['id'] == gcp_id]
        if filtered_df.empty:
            print(f"GCP ID {gcp_id} not found")
            return None
        
        self.lat = filtered_df.Y.values[0]
        self.long = filtered_df.X.values[0]
        self.elevation = filtered_df.Z.values[0]
        return (self.lat, self.long)
        
    def get_nearby_images(self, gcp_id):
        """Find images near a specific GCP"""        
        gcp_coords = self.get_gcp_coordinates(gcp_id)
        if not gcp_coords:
            return []

        candidates = self.image_finder.find_images_for_gcp(
            self.img_folder,
            (self.lat, self.long),
            self.elevation
        )

        if not candidates:
            return []

        return [candidate['path'] for candidate in candidates[:15]]

    def config_figure(self):
        """Configures the main matplotlip figure and its components for GCP identification"""        
        self.fig = figure(figsize=(14, 8))
        maximize_window(self.fig)
        width = self.current_image.shape[0]
        height = self.current_image.shape[1]
        
        # Title
        self.fig.text(0.5, 0.95, 'GCP Finder', fontsize=20, fontweight='bold', ha='center')
        
        # Main image axes
        self.ax = self.fig.add_subplot(111, xlim=(0,height), ylim=(0,width), autoscale_on=False)
        self.ax.set_position([0.20, 0.10, 0.63, 0.83])  
        
        # Configure y-axis ticks
        yticks = np.arange(0, width, 500)
        self.ax.set_yticks(yticks)
        self.ax.tick_params(axis='y', labelsize=8)
        self.ax.xaxis.set_ticks_position('bottom') # Show x-axis ticks at bottom
        
        # Initialize coordinate display
        self.coord_display = CoordDisplay(self.fig, position=[0.84, 0.15, 0.15, 0.80])
        
        # Instructions
        instructions = "Instruction:  Right-click at mouse to mark GCP location   |   Press 'D' to delete last point  |   Press 'S' to save points"
        instruction_ax = self.fig.add_axes([0.20, 0.02, 0.63, 0.04])
        instruction_ax.text(0.5, 0.5, instructions,
                        fontsize=10, va='center', ha='center',
                        transform=instruction_ax.transAxes)
        instruction_ax.set_xticks([])
        instruction_ax.set_yticks([])
        for spine in instruction_ax.spines.values():
            spine.set_visible(False)
        
        self.event_manager = EventManager(self) # Initialize event manager

    def setup_gcp_radio(self):
        """Create radio buttons for GCP selection showing only available IDs"""        
        rax = self.fig.add_axes([0.005, 0.52, 0.14, 0.36])
        gcp_tuple = tuple(sorted(self.gcp_df['id'].unique()))

        rax.set_title('GCP ID', pad=10, fontsize=15, fontweight='bold')
        rax.set_xticks([])
        rax.set_yticks([])
        for spine in rax.spines.values():
            spine.set_visible(False)
        
        self.radio = RadioButtons(rax, gcp_tuple)
        self.radio.on_clicked(self.setup_image_list_radio)

    def setup_image_list_radio(self, gcp_id):
        """Create radio buttons for image list or show no images message"""
        self.gcp_id = gcp_id

        # Remove existing axes
        for ax in self.fig.axes:
            if ax.get_position().bounds[0] == 0.005 and ax.get_position().bounds[1] == 0.10:
                self.fig.delaxes(ax)

        # Create new axes
        rax = self.fig.add_axes([0.005, 0.10, 0.14, 0.36])
        rax.set_title('Images with GCP', pad=10, fontsize=15, fontweight='bold')
        
        img_list = self.get_nearby_images(int(gcp_id)) # Get nearby images

        if not img_list:
            # Clear any existing radio buttons and main image
            if hasattr(self, 'radio2'): 
                del self.radio2
            
            self.ax.clear() # Clear main image display
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            rax.clear() # Display "No image found near GCP" in the radio button area
            rax.text(0.5, 0.5, 'No images found',
                    color='red',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=rax.transAxes,
                    fontsize=15)
            
            # Remove ticks and spines
            rax.set_xticks([])
            rax.set_yticks([])
            for spine in rax.spines.values():
                spine.set_visible(False)

            self.fig.canvas.draw_idle() # Force redraw
            return
        
        # If we have images, create the radio buttons
        img_names = [os.path.basename(i) for i in img_list]
        img_name_tuple = tuple(img_names[:15])
        
        if img_name_tuple:
            self.radio2 = RadioButtons(rax, img_name_tuple)
            rax.set_xticks([])
            rax.set_yticks([])
            for spine in rax.spines.values():
                spine.set_visible(False)
            self.radio2.on_clicked(self.load_and_display_image)
            if img_name_tuple:
                self.load_and_display_image(img_name_tuple[0])

    def save_collected_points(self):
        """Save collected points with proper coordinate handling"""
        if not self.collected_points:
            self.show_save_message("No points to save!", color='red')
            return

        output_file = os.path.join(self.img_folder, 'gcp_list.txt')
        with open(output_file, 'w') as f:
            # Write header
            f.write(f"WGS84 UTM {self.coord_transformer.utm_zone}{self.coord_transformer.hemisphere}\n")
            
            # Sort and group points
            sorted_points = sorted(self.collected_points, key=lambda x: x['id'])
            grouped_points = groupby(sorted_points, key=lambda x: x['id'])
            
            # Write points
            for gcp_id, points in grouped_points:
                points = list(points)
                gcp_info = self.gcp_df[self.gcp_df['id'] == int(gcp_id)].iloc[0] # Get GCP from dataframe
                utm_x, utm_y = self.coord_transformer.transform_coordinates(
                    gcp_info['Y'], # Latitude
                    gcp_info['X']  # Longitude
                )
                
                if utm_x is None or utm_y is None:
                    continue
                    
                elev = gcp_info['Z']
                
                for point in points:
                    f.write(f"{utm_x:.9f} {utm_y:.9f} {elev:.2f} "
                           f"{point['x']:.13f} {point['y']:.13f} "
                           f"{point['fname']} {point['id']}\n")
                    
        self.show_save_message(f"Points saved to:\n{os.path.basename(self.img_folder)}/gcp_list.txt")

    def show_save_message(self, message, color='green'):
        """Display a temporary message on the plot"""
        # Create message box
        message_box = self.fig.add_axes([0.35, 0.45, 0.3, 0.1]) 
        message_box.set_facecolor('white')
        message_box.set_alpha(0.9)

        # Add text
        message_box.text(0.5, 0.5, message, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=message_box.transAxes,
                        color=color,
                        fontsize=12,
                        fontweight='bold')
        
        # Configure box appearance
        message_box.set_xticks([])
        message_box.set_yticks([])
        for spine in message_box.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        
        self.fig.canvas.draw()
        plt.pause(2) # Show message for 2 seconds
        
        self.fig.delaxes(message_box)
        self.fig.canvas.draw()

    def start(self):
        """Initialize and start the GCP identification process"""
        self.config_figure()
        self.setup_gcp_radio()
        self.setup_image_list_radio('1')
        show()
        
def main():
    """Main program execution"""
    initial_window = InitialWindow()
    initial_window.show()

if __name__ == "__main__":
    main()     