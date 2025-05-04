"""
Script Name: stat_extractor_mac.py
Purpose: To extract spatial statistics from raster data using CSV (bounding boxes) with transformation parameters
Date Created: March 15, 2025
Version: 3.0
"""

import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'  # Silence deprecation warning
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use('MacOSX')  # For MacOSX backend 
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import subprocess
import threading
import time
from shapely.geometry import box
from shapely.geometry import Polygon

def maximize_window(fig):
    """Set the figure to a reasonable size"""
    fig.set_size_inches(14, 8, forward=True)

class FileDialogHandler:
    """Handler for file dialogs using macOS native dialogs"""
    @staticmethod
    def ask_open_filename(title='Select File', filetypes=None):
        cmd = (
            'osascript -e \'tell application "SystemUIServer"\n'
            'activate\n'
            'set filePath to POSIX path of (choose file with prompt "' + title + '")\n'
            'end tell\''
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    
    @staticmethod
    def ask_save_filename(title='Save As', filetypes=None, defaultextension=None):
        default_name = "output.csv" if defaultextension == ".csv" else "output.txt"
        cmd = (
            'osascript -e \'tell application "SystemUIServer"\n'
            'activate\n'
            'set filePath to POSIX path of (choose file name with prompt "' + title + '" default name "' + default_name + '")\n'
            'end tell\''
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
        
    @staticmethod
    def ask_directory(title='Select Directory'):
        cmd = (
            'osascript -e \'tell application "SystemUIServer"\n'
            'activate\n'
            'set folderPath to POSIX path of (choose folder with prompt "' + title + '")\n'
            'end tell\''
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

def csv_to_geopandas(bounds_df, raster_path=None, manual_crs=None):
    """Convert bounds CSV data to a GeoDataFrame using CRS from:
    1. User input (if provided)
    2. Raster image (if available)
    """
    global box_geometry
    box_geometry = []
        
    # Determine CRS with priority: user input > raster
    crs = None
    crs_source = None
    
    # 1. Try manual CRS first if provided
    if manual_crs and manual_crs.strip():
        # Convert string like '4326' to integer if needed
        if isinstance(manual_crs, str) and manual_crs.isdigit():
            manual_crs = int(manual_crs)
                
        # Create CRS from user input
        from pyproj.crs import CRS
        crs = CRS.from_user_input(manual_crs)
        crs_source = "user"
        
    # 2. Try raster CRS if user input not available or invalid
    if crs is None and raster_path:
        with rasterio.open(raster_path) as src:
            crs = src.crs
            if crs:
                crs_source = "raster"
        
    # If no CRS could be determined, return None
    if crs is None:
        return None
        
    # Process each row to create geometries
    for idx, row in bounds_df.iterrows():
        # Extract transform parameters
        transform = rasterio.transform.Affine(*row[0:6].values)
        transformer = rasterio.transform.AffineTransformer(transform)
            
        # Get upper left corner
        ul = transformer.xy(0, 0)     
        # Parse column 9
        lr_y = int(row['9'].split(',')[-1][1:-1])
        lr_x = int(row['9'].split(',')[1][1:])
        lr_pixels = [lr_x, lr_y]
        lr = transformer.xy(*lr_pixels)
            
        # Create box
        box_lim = [ul[0], ul[1], lr[0], lr[1]]
        bounding_polygon = box(box_lim[0], box_lim[1], box_lim[2], box_lim[3])
        box_geometry.append(Polygon(bounding_polygon))
        
    # Create a GeoDataFrame with determined CRS
    if box_geometry:
        gdf = gpd.GeoDataFrame(geometry=box_geometry, crs=crs)
        gdf['id'] = range(1, len(gdf) + 1)
        return gdf
    else:
        return None
    
class CombinedPreviewCanvas:
    """Canvas for displaying combined raster and bounds preview"""
    def __init__(self, fig, position):
        self.ax = fig.add_axes(position)
        self.ax.set_title('Combined Preview')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.show_message("No data loaded")
        self.raster_loaded = False
        self.bounds_loaded = False
        self.raster_path = None
        self.bounds_df = None
        self.bounds_gdf = None
        self.raster_img = None
        self.fig = fig
        
    def show_message(self, message):
        """Display a message on the plot"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.draw()
        
    def load_raster(self, raster_path):
        """Load raster data"""
        # Update state
        self.raster_path = raster_path
        self.raster_loaded = True
        
        # Clear current axes and any existing image references
        self.ax.clear()
        if hasattr(self, 'raster_img') and self.raster_img:
            self.raster_img = None
            
        # Display raster with optimized memory usage
        self.show_raster_only()
        
        # If bounds are also loaded, update with combined view
        if self.bounds_loaded:
            self.show_combined()

    def load_bounds(self, bounds_df):
        """Load bounds data and convert to shapefile"""
        self.bounds_df = bounds_df
        
        # Convert CSV to GeoDataFrame (shapefile format)
        self.bounds_gdf = csv_to_geopandas(
            bounds_df,
            self.raster_path if self.raster_loaded else None,
            getattr(self, 'manual_crs', None)  # Safely get manual_crs if it exists
        )
        
        if self.bounds_gdf is not None:
            self.bounds_loaded = True
            
            # Update the preview with both layers if raster is also loaded
            if self.raster_loaded:
                self.show_combined()
            else:
                self.show_bounds_only()
        else:
            self.show_message("Failed to convert bounds to shapefile format")
    
    def show_raster_only(self):
        """Plot raster preview only"""
        # Open and read raster
        with rasterio.open(self.raster_path) as src:
            # Number of bands will determine plotting approach
            num_bands = src.count
            
            if num_bands == 1:
                # Single band - grayscale or pseudocolor
                raster_data = src.read(1)
                
                # Handle nodata values
                if src.nodata is not None:
                    mask = raster_data == src.nodata
                    raster_data = np.ma.array(raster_data, mask=mask)
                
                # Display raster with viridis colormap
                self.raster_img = self.ax.imshow(raster_data, cmap='viridis')
            
            elif num_bands >= 3:
                # RGB composite (natural color)
                rgb_bands = [1, 2, 3]  # Assuming 1=Red, 2=Green, 3=Blue
                
                # Read the bands
                rgb = np.zeros((src.height, src.width, 3), dtype=np.uint8)
                for i, band in enumerate(rgb_bands):
                    # Only read if band exists
                    if band <= num_bands:
                        band_data = src.read(band)
                        # Normalize to 0-255 for display
                        band_min, band_max = np.percentile(band_data[band_data > 0], (2, 98))
                        rgb[:, :, i] = np.clip(255 * (band_data - band_min) / (band_max - band_min), 0, 255).astype(np.uint8)
                
                # Display RGB composite
                self.raster_img = self.ax.imshow(rgb)
            
            # Set title
            self.ax.set_title(f"Raster: {os.path.basename(self.raster_path)}")
        
        plt.draw()
    
    def show_bounds_only(self):
        """Plot bounds only as a shapefile"""
        # Check if bounds data is available
        if self.bounds_gdf is None or len(self.bounds_gdf) == 0:
            self.show_message("No bounds data available")
            return
        
        # Plot the geometries
        self.bounds_gdf.plot(ax=self.ax, edgecolor='blue', facecolor='none', linewidth=1.0)
        
        # Add labels (box IDs)
        for idx, row in self.bounds_gdf.iterrows():
            # Get the centroid for label placement
            centroid = row['geometry'].centroid
            self.ax.text(centroid.x, centroid.y, str(row['id']), 
                       fontsize=8, ha='center', va='center')
        
        # Set title
        self.ax.set_title(f"Bounds Preview ({len(self.bounds_gdf)} boxes)")
        
        plt.draw()
    
    def show_combined(self):
        """Show both raster and bounds as shapefile overlay"""
        # Clear previous content
        self.ax.clear()
        
        # Show loading message
        plt.draw()
        plt.pause(0.1)  
        
        # Load and display the raster data
        with rasterio.open(self.raster_path) as src:
            # Get image dimensions and bounds
            height = src.height
            width = src.width
            bounds = src.bounds
            
            # For large rasters, downsample
            if max(height, width) > 3000:
                factor = max(height, width) / 1500
                img = src.read(
                    out_shape=(
                        src.count,
                        int(height / factor),
                        int(width / factor)
                    ),
                    resampling=rasterio.enums.Resampling.average
                )
                transform = src.transform * src.transform.scale(
                    (width / img.shape[-1]),
                    (height / img.shape[-2])
                )
            else:
                img = src.read()
                transform = src.transform
            
            # Display the raster with natural colors
            if src.count == 1:
                # For single band
                raster_img = img[0]
                # Use straightforward min-max normalization for more natural look
                valid_data = raster_img[~np.isnan(raster_img) & (raster_img != src.nodata)]
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, (1, 99))
                    self.ax.imshow(
                        raster_img, 
                        cmap='gray',
                        vmin=vmin,
                        vmax=vmax,
                        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                        interpolation='bilinear'
                    )
            else:
                # For RGB, use natural colors without enhancement
                rgb = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
                for i in range(min(3, src.count)):
                    band = img[i]
                    # Use gentle normalization
                    valid_band = band[~np.isnan(band) & (band != src.nodata if src.nodata is not None else True)]
                    if len(valid_band) > 0:
                        vmin, vmax = np.percentile(valid_band, (1, 99))
                        normalized = np.clip(255 * (band - vmin) / (vmax - vmin), 0, 255).astype(np.uint8)
                        rgb[:, :, i] = normalized
                
                # Display the RGB composite with natural colors
                self.ax.imshow(
                    rgb,
                    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                    interpolation='bilinear'
                )
            
            # Update UI
            plt.draw()
            plt.pause(0.1)
            
            # Render boundaries
            if self.bounds_gdf is not None and len(self.bounds_gdf) > 0:
                # Use semi-transparent boundaries with thinner lines
                self.bounds_gdf.plot(
                    ax=self.ax,
                    facecolor="none",
                    edgecolor=(1, 0, 0, 0.5),
                    linewidth=0.8,
                    linestyle='-',
                    zorder=10
                )
            
            # Set view limits
            self.ax.set_xlim(bounds.left, bounds.right)
            self.ax.set_ylim(bounds.bottom, bounds.top)
            
            # Title and layout
            self.ax.set_title(f"Preview: Shapefile layer and {os.path.basename(self.raster_path)}")
            self.ax.set_aspect('equal')
            self.fig.tight_layout()

        plt.draw()

    def cleanup_resources(self):
        """Release memory used by visualization"""
        # Clear references to large objects
        if hasattr(self, 'raster_img') and self.raster_img:
            self.raster_img = None
        
        # Force garbage collection
        import gc
        gc.collect()

class ZonalStatsWorker(threading.Thread):
    """Worker thread for running zonal statistics calculations"""
    def __init__(self, bounds_file, raster_file, output_file, stats_to_calculate, callback):
        super().__init__()
        self.bounds_file = bounds_file
        self.raster_file = raster_file
        self.output_file = output_file
        self.stats_to_calculate = stats_to_calculate
        self.callback = callback
        self.progress = 0
        self.result = None
        self.error = None
        self.percentile_value = 90  # Default value
            
    def run(self):
        # Emit initial progress
        self.progress = 5
        
        # Load bounds file
        bounds_df = pd.read_csv(self.bounds_file)
        self.progress = 15
        
        # Load raster
        with rasterio.open(self.raster_file) as src:
            raster_data = src.read(1)
            
            # Transform bounds to pixel coordinates
            box_lim_list, box_lim_pixel_list = self.find_box_limits(bounds_df, src)
            self.progress = 40
            
            # Calculate statistics for each box 
            results = []
            for i, (box_lim, box_pixels) in enumerate(zip(box_lim_list, box_lim_pixel_list)):
                # Update progress proportionally
                self.progress = 40 + int((i / len(box_lim_list)) * 50)
                
                # Get pixel values within the box
                ul_row, ul_col = box_pixels[0], box_pixels[1]
                lr_row, lr_col = box_pixels[2], box_pixels[3]
                
                # Ensure the indices are in bounds and correctly ordered
                min_row, max_row = min(ul_row, lr_row), max(ul_row, lr_row)
                min_col, max_col = min(ul_col, lr_col), max(ul_col, lr_col)
                
                # Get pixel values within the box
                pixel_values = raster_data[min_row:max_row+1, min_col:max_col+1]
                pixel_values = pixel_values[~np.isnan(pixel_values)]
                
                # Calculate statistics
                stats = {}
                if 'min' in self.stats_to_calculate and len(pixel_values) > 0:
                    stats['min'] = np.nanmin(pixel_values)
                if 'max' in self.stats_to_calculate and len(pixel_values) > 0:
                    stats['max'] = np.nanmax(pixel_values)
                if 'mean' in self.stats_to_calculate and len(pixel_values) > 0:
                    stats['mean'] = np.nanmean(pixel_values)
                if 'median' in self.stats_to_calculate and len(pixel_values) > 0:
                    stats['median'] = np.nanmedian(pixel_values)
                if 'std' in self.stats_to_calculate and len(pixel_values) > 0:
                    stats['std'] = np.nanstd(pixel_values)
                if 'sum' in self.stats_to_calculate and len(pixel_values) > 0:
                    stats['sum'] = np.nansum(pixel_values)
                if 'count' in self.stats_to_calculate:
                    stats['count'] = len(pixel_values)

                # Calculate percentile if selected
                if 'percentile' in self.stats_to_calculate and len(pixel_values) > 0:
                    stats[f'percentile_{self.percentile_value}'] = np.nanpercentile(pixel_values, self.percentile_value)
                   
                # Add box info
                stats['box_id'] = i + 1
                stats['ul_x'], stats['ul_y'] = box_lim[0], box_lim[1]
                stats['lr_x'], stats['lr_y'] = box_lim[2], box_lim[3]
                
                results.append(stats)
        
        # Update progress to 95% before creating DataFrame
        self.progress = 95
        # Create a DataFrame from results
        results_df = pd.DataFrame(results)
        # Update progress to 98% before saving file
        self.progress = 98

        # Save results automatically to CSV
        results_df.to_csv(self.output_file, index=False)
        
        # Ensure progress reaches 100% before completion
        self.progress = 100
        # Add a small pause to ensure UI can update before thread exits
        time.sleep(0.2)
        # Store the result
        self.result = results_df
        
        # Call the callback to signal completion
        if self.callback:
            self.callback(self)
            
    def find_box_limits(self, box_df, ortho_dataset):
        """Find the geographical and pixel bounds for each box"""
        box_lim_list = []
        box_lim_pixel_list = []
        
        for box_row_index in range(len(box_df)):
            bounds1 = box_df.iloc[box_row_index]
            transform1 = rasterio.transform.Affine(*bounds1[0:6].values)
            transformer1 = rasterio.transform.AffineTransformer(transform1)
            ul = transformer1.xy(0, 0)
            
            # Fixed access to column 9 in CSV
            col9_str = str(bounds1.iloc[9] if isinstance(bounds1.index, pd.RangeIndex) else bounds1['9'])
            lr_y = int(col9_str.split(',')[-1][1:-1])
            lr_x = int(col9_str.split(',')[1][1:])
            lr_pixels = [lr_x, lr_y]
            lr = transformer1.xy(*lr_pixels)
            
            box_lim = [*ul, *lr]
            
            ul_pixels = ortho_dataset.index(*ul)
            lr_pixels = ortho_dataset.index(*lr)
            
            box_lim_pixels = [*ul_pixels, *lr_pixels]
            box_lim_list.append(box_lim)
            box_lim_pixel_list.append(box_lim_pixels)
                
        return box_lim_list, box_lim_pixel_list

class SpatialStatsExtractorGUI:
    """Main GUI for spatial statistics extraction with matplotlib"""
    def __init__(self):
        # Initialize properties
        self.bounds_path = None
        self.bounds_df = None
        self.raster_path = None
        self.output_path = None
        self.results_df = None
        self.worker = None
        self.file_dialog = FileDialogHandler()
        
        # Create figure and UI
        self.fig = plt.figure(figsize=(14, 8))
        maximize_window(self.fig)
        
        # Add border to the figure
        self.fig.patch.set_edgecolor('black')
        self.fig.patch.set_linewidth(1)
        
        # Setup UI components
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the main UI components"""
        # Main title
        self.fig.suptitle('Spatial Statistics Extractor', fontsize=20, fontweight='bold', y=0.98)
        
        # Create layouts for each section
        self._setup_left_panel()
        self._setup_preview_panel()
        self._setup_status_section()
        
    def _setup_left_panel(self):
        """Setup the left panel with inputs and stats"""
        # Input Configuration Section
        self.fig.text(0.15, 0.90, "Input Configuration", fontsize=14, fontweight='bold', ha='center')
        
        # CSV input
        self.fig.text(0.04, 0.85, "CSV File:", ha='left')
        bounds_ax = self.fig.add_axes([0.04, 0.81, 0.21, 0.03])
        self.bounds_input = TextBox(bounds_ax, '', initial='')
        
        bounds_button_ax = self.fig.add_axes([0.26, 0.81, 0.08, 0.03])
        self.bounds_button = Button(bounds_button_ax, 'Browse')
        self.bounds_button.on_clicked(self.browse_bounds)
        
        # Raster file input
        self.fig.text(0.04, 0.78, "Raster File:", ha='left')
        raster_ax = self.fig.add_axes([0.04, 0.74, 0.21, 0.03])
        self.raster_input = TextBox(raster_ax, '', initial='')
        
        raster_button_ax = self.fig.add_axes([0.26, 0.74, 0.08, 0.03])
        self.raster_button = Button(raster_button_ax, 'Browse')
        self.raster_button.on_clicked(self.browse_raster)

        # CRS input
        self.fig.text(0.04, 0.71, "CRS EPSG Code: ", ha='left')
        crs_ax = self.fig.add_axes([0.04, 0.67, 0.21, 0.03])
        self.crs_input = TextBox(crs_ax, '', initial='')  # Empty by default
        self.fig.text(0.26, 0.68, "(optional)", ha='left', fontsize=10, color='gray')
        
        # Add a help note about CRS
        crs_note_ax = self.fig.add_axes([0.04, 0.655, 0.30, 0.025])
        crs_note_ax.axis('off')
        crs_note_ax.text(0, 0, "Leave empty to use CRS from raster", fontsize=8, color='gray')

        # Preview button
        preview_button_ax = self.fig.add_axes([0.04, 0.60, 0.30, 0.035])
        self.preview_button = Button(preview_button_ax, 'Click to Update Preview', color='#90EE90', hovercolor='#7CCD7C')
        self.preview_button.on_clicked(self.update_preview)

        # Output Configuration Section
        self.fig.text(0.15, 0.55, "Output Configuration", fontsize=14, fontweight='bold', ha='center')
        
        # Statistics Section
        self.fig.text(0.04, 0.505, "Statistics to Calculate", ha='left')
        
        # Statistics checkboxes - manual implementation
        self.stat_buttons = {}
        stats = [
            ('min', 'Minimum'), 
            ('max', 'Maximum'), 
            ('mean', 'Mean'),
            ('median', 'Median'),
            ('std', 'Standard Deviation'),
            ('sum', 'Sum'),
            ('count', 'Count'),
            ('percentile', 'Percentile')
        ]
        
        # Create a frame for the checkboxes
        check_ax = self.fig.add_axes([0.05, 0.35, 0.29, 0.18])
        check_ax.axis('off')
        
        # Create percentile input box BEFORE creating checkboxes and their callbacks
        # Add percentile input box (initially hidden)
        percentile_y_pos = 0.47 - (len(stats) - 1) * 0.025  # Position it next to the percentile checkbox
        
        # Create label for percentile value
        self.percentile_label_ax = self.fig.add_axes([0.18, percentile_y_pos-0.005, 0.05, 0.02])
        self.percentile_label_ax.axis('off')
        self.percentile_label_ax.text(-0.5, 0.5, "Value:", fontsize=10, ha='left', va='center')
        self.percentile_label_ax.set_visible(False)  # Initially hidden
        
        # Create textbox for percentile value input
        self.percentile_input_ax = self.fig.add_axes([0.19, percentile_y_pos-0.005, 0.035, 0.025])
        self.percentile_textbox = TextBox(self.percentile_input_ax, '', initial='90')
        self.percentile_input_ax.set_visible(False)  # Initially hidden
        
        # Create each checkbox as a separate button with checkmark symbol
        for i, (code, name) in enumerate(stats):
            y_pos = 0.47 - (i * 0.025)
            
            # Checkbox area
            check_box_ax = self.fig.add_axes([0.07, y_pos-0.003, 0.02, 0.02])
            check_box = Button(check_box_ax, '', color='white')
            
            # Default state - all unchecked
            initial_state = False
            
            # Store button state
            check_box.code = code
            check_box.name = name
            check_box.state = initial_state
            
            # Create the callback to toggle checkmark instead of color
            def make_callback(btn):
                def callback(event):
                    btn.state = not btn.state
                    btn.label.set_text('✓' if btn.state else '')
                    
                    # Handle percentile input visibility if this is the percentile checkbox
                    if btn.code == 'percentile':
                        if btn.state:
                            # Show percentile input
                            self.percentile_label_ax.set_visible(True)
                            self.percentile_input_ax.set_visible(True)
                        else:
                            # Hide percentile input
                            self.percentile_label_ax.set_visible(False)
                            self.percentile_input_ax.set_visible(False)
                    
                    self.fig.canvas.draw_idle()
                return callback
            
            check_box.on_clicked(make_callback(check_box))
            self.stat_buttons[code] = check_box
            
            # Label for checkbox
            self.fig.text(0.10, y_pos + 0.005, name, ha='left', va='center', fontsize=10)

        # Output file
        self.fig.text(0.04, 0.255, "Save Output File:", ha='left')
        output_ax = self.fig.add_axes([0.04, 0.215, 0.21, 0.03])
        self.output_input = TextBox(output_ax, '', initial='')
        
        output_button_ax = self.fig.add_axes([0.26, 0.215, 0.08, 0.03])
        self.output_button = Button(output_button_ax, 'Browse')
        self.output_button.on_clicked(self.browse_output)
        
        # Calculate button
        calc_ax = self.fig.add_axes([0.04, 0.14, 0.30, 0.06])
        self.calc_button = Button(calc_ax, 'Click to Calculate Statistics and Save File', color='#90EE90', hovercolor='#7CCD7C')
        self.calc_button.on_clicked(self.calculate_statistics)
        
        # Initially disable buttons
        self.calc_button.set_active(False)
        self.calc_button.color = '#cccccc'
        self.preview_button.set_active(False)
        self.preview_button.color = '#cccccc'

    def _setup_preview_panel(self):
        """Setup the preview panel on the right"""
        # Create a combined preview canvas
        self.preview_canvas = CombinedPreviewCanvas(self.fig, [0.42, 0.15, 0.55, 0.75])
    
    def _setup_status_section(self):
        """Setup status and progress section"""
        # Status message
        self.status_ax = self.fig.add_axes([0.05, 0.05, 0.29, 0.08])
        self.status_ax.set_xticks([])
        self.status_ax.set_yticks([])
        for spine in self.status_ax.spines.values():
            spine.set_visible(False)
        
        self.show_status("Ready to start. Please select input files.")
        
        # Progress bar (as a rectangle)
        self.progress_ax = self.fig.add_axes([0.05, 0.02, 0.25, 0.02]) 
        self.progress_bar = self.progress_ax.barh(0, 0, color='#4CAF50', height=0.5)
        self.progress_ax.set_xlim(0, 100)
        self.progress_ax.set_xticks([])
        self.progress_ax.set_yticks([])
        self.progress_ax.set_visible(False)
        
        # Add percentage text
        self.percent_ax = self.fig.add_axes([0.31, 0.02, 0.03, 0.02])
        self.percent_ax.axis('off')
        self.percent_text = self.percent_ax.text(0, 0.5, "0%", fontsize=10, ha='left', va='center')
        self.percent_ax.set_visible(False)
    
    def browse_bounds(self, event):
        """Open file dialog to select bounds CSV file"""
        file_path = self.file_dialog.ask_open_filename(
            title="Select Bounds CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.bounds_path = file_path
            self.bounds_input.set_val(os.path.basename(file_path))
            self.show_status(f"Loading {os.path.basename(file_path)}...")
            
            # Load CSV
            self.bounds_df = pd.read_csv(file_path)           
            # Show status
            self.show_status(f"Loaded {os.path.basename(file_path)}: {len(self.bounds_df)} bounding boxes")
            # Enable preview button if raster is also loaded
            self._check_inputs()
    
    def browse_raster(self, event):
        """Open file dialog to select raster file"""
        file_path = self.file_dialog.ask_open_filename(
            title="Select Raster File",
            filetypes=[("Raster Files", "*.tif *.tiff *.img"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.raster_path = file_path
            self.raster_input.set_val(os.path.basename(file_path))
            self.show_status(f"Loading {os.path.basename(file_path)}...")
            
            # Check if raster has CRS info
            with rasterio.open(file_path) as src:
                crs = src.crs
                if crs:
                    self.show_status(f"Loaded {os.path.basename(file_path)} with CRS: {crs}")
                else:
                    self.show_status(f"Loaded {os.path.basename(file_path)} - WARNING: No CRS found in raster.")
            
            # Enable preview button if bounds are also loaded
            self._check_inputs()
    
    def browse_output(self, event):
        """Open file dialog to select output file"""
        file_path = self.file_dialog.ask_save_filename(
            title="Select Output CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            defaultextension=".csv"
        )
        
        if file_path:
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
                
            self.output_path = file_path
            self.output_input.set_val(os.path.basename(file_path))
            
            # Check if we can enable buttons
            self._check_inputs()
    
    def _check_inputs(self):
        """Check if all required inputs are provided"""
        has_bounds = self.bounds_path is not None and os.path.exists(self.bounds_path)
        has_raster = self.raster_path is not None and os.path.exists(self.raster_path)
        has_output = self.output_path is not None
        
        # Enable preview button if both bounds and raster are loaded
        if has_bounds and has_raster:
            self.preview_button.color = '#ADD8E6'  # Light blue
            self.preview_button.set_active(True)
        else:
            self.preview_button.color = '#cccccc'  # Disabled gray
            self.preview_button.set_active(False)
        
        # Enable calculate button if all inputs are provided
        if has_bounds and has_raster and has_output:
            self.calc_button.color = '#4CAF50'  # Bright green
            self.calc_button.set_active(True)
        else:
            self.calc_button.color = '#cccccc'  # Disabled gray
            self.calc_button.set_active(False)
        
        self.fig.canvas.draw_idle()
    
    def update_preview(self, event=None):
        """Update the combined preview"""
        if not self.bounds_path or not self.raster_path:
            self.show_status("Missing input files for preview")
            return
            
        # Disable preview button during processing
        self.preview_button.set_active(False) 
        self.preview_button.color = '#cccccc'
        self.fig.canvas.draw_idle()
        
        # Get manual CRS if provided by user
        manual_crs = self.crs_input.text.strip() if hasattr(self, 'crs_input') else None
        
        # Pass the manual CRS to preview canvas
        self.preview_canvas.manual_crs = manual_crs
        
        # Load the raster first
        self.show_status("Loading raster...")
        self.preview_canvas.load_raster(self.raster_path)
        
        # Update status for bounds loading
        crs_msg = f" with user-specified CRS: {manual_crs}" if manual_crs else " with CRS from raster"
        self.show_status(f"Loading bounds{crs_msg}...")
        
        # Load bounds
        self.preview_canvas.load_bounds(self.bounds_df)
        
        # Update status based on results
        if not self.preview_canvas.bounds_loaded:
            self.show_status("Failed to load bounds: No valid CRS. Please specify a CRS code.")
        else:
            # Get CRS info for status message
            crs_used = self.preview_canvas.bounds_gdf.crs
            self.show_status(f"Preview updated with both layers")
        
        # Re-enable the preview button
        self.preview_button.set_active(True)
        self.preview_button.color = '#ADD8E6'
        self.fig.canvas.draw_idle()
    
    def calculate_statistics(self, event):
        """Start spatial statistics calculation"""
        # Collect selected statistics
        selected_stats = [code for code, button in self.stat_buttons.items() if button.state]
        
        if not selected_stats:
            self.show_status("Please select at least one statistic to calculate")
            return
        
        # Get percentile value if percentile stat is selected
        percentile_value = 90  # Default value
        if 'percentile' in selected_stats:
            try:
                percentile_value = float(self.percentile_textbox.text)
                if percentile_value < 0 or percentile_value > 100:
                    self.show_status("Percentile value must be between 0 and 100. Using default value of 90.")
                    percentile_value = 90
            except ValueError:
                self.show_status("Invalid percentile value. Using default value of 90.")
                percentile_value = 90
        
        # Update UI
        self.calc_button.set_active(False)
        self.calc_button.color = '#cccccc'
        self.progress_ax.set_visible(True)
        self.percent_ax.set_visible(True)
        self.percent_text.set_text("0%")
        self.show_status("Calculating spatial statistics...")
        
        # Start worker thread
        self.worker = ZonalStatsWorker(
            self.bounds_path,
            self.raster_path,
            self.output_path,
            selected_stats,
            self.on_calculation_complete
        )
        
        # Add percentile value to worker
        self.worker.percentile_value = percentile_value
        
        # Start a timer to update the progress
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.update_progress_from_worker)
        self.timer.start()
        
        # Start worker
        self.worker.start()
    
    def update_progress_from_worker(self):
        """Update progress from worker thread"""
        if self.worker:
            # Update progress bar
            progress_value = self.worker.progress
            self.progress_bar[0].set_width(progress_value)
            
            # Update percentage text
            self.percent_text.set_text(f"{progress_value}%")
            
            # Force redraw for smoother updates
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
    
    def on_calculation_complete(self, worker):
        """Handle calculation completion"""
        # Stop the timer
        self.timer.stop()
        
        # Check for errors
        if worker.error:
            self.show_status(worker.error)
            self.progress_ax.set_visible(False)
            self.percent_ax.set_visible(False)
            self.calc_button.set_active(True)
            self.calc_button.color = '#4CAF50'
            self.fig.canvas.draw_idle()
            return
        
        # Store results
        self.results_df = worker.result
        
        # Update status
        stats_count = len(self.results_df) if self.results_df is not None else 0
        self.show_status(f"Calculation complete! {stats_count} results saved to {os.path.basename(self.output_path)}")
        
        # Re-enable calculate button
        self.calc_button.set_active(True)
        self.calc_button.color = '#4CAF50'
        
        # Hide progress bar and percentage
        self.progress_ax.set_visible(False)
        self.percent_ax.set_visible(False)
        self.fig.canvas.draw_idle()
    
    def show_status(self, message):
        """Show a status message"""
        # Clear previous text
        self.status_ax.clear()
        self.status_ax.set_xticks([])
        self.status_ax.set_yticks([])
        for spine in self.status_ax.spines.values():
            spine.set_visible(False)
        
        # Show new message with larger red text
        self.status_ax.text(0.0, 0.5, message, ha='left', va='center', 
                          fontsize=12, color='red', fontweight='bold')
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the GUI window"""
        plt.show()

def main():
    """Run the Mac application"""
    # Create and show GUI
    gui = SpatialStatsExtractorGUI()
    gui.show()

if __name__ == "__main__":
    main()