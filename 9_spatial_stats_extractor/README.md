# **Spatial Statistics Extractor Tool**

A Python utility for extracting spatial statistics from raster data using CSV bounding boxes with transformation parameters. This tool provides an interactive interface for analyzing raster data within specified regions and calculating various statistical metrics.

### **Note**

This tool is essential for analyzing spatial data in precision agriculture, environmental monitoring, and remote sensing workflows. It efficiently extracts statistical information from defined regions of interest in geospatial raster data, supporting better decision-making and data analysis.

## Table of Contents
- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Input Requirements](#input-requirements)
- [Outputs](#outputs)
- [Usage Instructions](#usage-instructions)
- [Interactive GUI Controls](#interactive-gui-controls)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [License](#license)

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/username/repo-name.git
cd repo-name/spatial_statistics_extractor
```

### 2. Install dependencies:

- For macOS:
  ```bash
  pip install -r requirements_mac.txt
  ```

- For Windows:
  ```bash
  pip install -r requirements_win.txt
  ```

### 3. Run the script:

- For macOS:
  ```bash
  python stat_extractor_mac.py
  ```

- For Windows:
  ```bash
  python stat_extractor_win.py
  ```

## **Features**

- Interactive GUI: User interface for statistical analysis
- Raster Visualization: Preview raster data and bounding boxes
- Multiple Statistics: Calculate min, max, mean, median, standard deviation, sum, count, and percentiles
- Coordinate System Support: Handle different coordinate reference systems
- Multi-threaded Processing: Efficient calculation for large datasets
- Progress Tracking: Visual progress indicators during calculations
- CSV Output: Export results in organized CSV format

## **Requirements**

- Python 3.9+
- Dependencies:
  - NumPy
  - Pandas
  - GeoPandas
  - Rasterio
  - Matplotlib
  - Shapely
  - PyProj

## **Input Requirements**

### 1. Required Files
- Raster File:
   - Format: GeoTIFF (.tif/.tiff) or other raster formats supported by Rasterio
   - Must contain geospatial reference information (CRS)
- CSV Bounds File:
   - Contains transformation parameters for bounding boxes
   - This file will come from the output of module 8 of the processing pipeline
   - Format requires 11 columns with the following structure:
     - Columns 0-5: Affine transformation parameters
     - Columns 6-8: Additional parameters
     - Column 9: String containing pixel dimensions for the bounding box
     - Column 10: file name
   - Example of CSV bounds file structure:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1153 | -0.0012 | 605805.5060 | 0.0012 | 0.1153 | 3766997.0630 | 0.0000 | 0.0000 | 0.0000 | "(0, 0), (952, 705)" | s_190529_135201_0_R01_3 |
| 0.1153 | -0.0019 | 605805.9583 | 0.0019 | 0.1153 | 3766881.9673 | 0.0000 | 0.0000 | 0.0000 | "(0, 0), (952, 701)" | s_190529_135201_0_R02_3 |
| 0.1153 | -0.0005 | 605805.1830 | 0.0005 | 0.1153 | 3767112.6523 | 0.0000 | 0.0000 | 0.0000 | "(0, 0), (952, 704)" | s_190529_135201_0_R00_3 |

### 2. Configuration Setup
- CRS Configuration:
   - Optional EPSG code input
   - Automatic CRS detection from raster if not specified
- Statistics Selection:
   - Choose which statistics to calculate (min, max, mean, etc.)
   - Customizable percentile values

## **Outputs**

- CSV File:
   - Contains calculated statistics for each bounding box
   - Column order: box_id, ul_x, ul_y, lr_x, lr_y, followed by statistical values
   - Example output format:

| box_id | ul_x | ul_y | lr_x | lr_y | min | max | mean | median | std | sum | count | percentile_90 |
|--------|------|------|------|------|-----|-----|------|--------|-----|------|-------|--------------|
| 1 | 605805.5060 | 3766997.0630 | 605914.8966 | 3766879.7401 | 0.0156 | 0.1938 | 0.0837 | 0.0812 | 0.0421 | 6124.83 | 73182 | 0.1412 |
| 2 | 605805.9583 | 3766881.9673 | 605915.8652 | 3766764.8903 | 0.0203 | 0.2104 | 0.0924 | 0.0897 | 0.0458 | 6182.13 | 66912 | 0.1535 |
| 3 | 605805.1830 | 3767112.6523 | 605914.9826 | 3766995.1083 | 0.0187 | 0.1867 | 0.0798 | 0.0774 | 0.0399 | 5352.98 | 67069 | 0.1352 |

## **Usage Instructions**

- Launch Application:
   ```python
   python stat_extractor_mac.py  # or stat_extractor_win.py for Windows
   ```
- Initial Setup:
   - Click "Browse" to select your CSV bounds file
   - Click "Browse" to select your raster file
   - Optionally enter a CRS EPSG code if not using the raster's CRS
   - Click "Click to Update Preview" to visualize data
  
- Statistics Configuration:
   - Check the statistics you want to calculate
   - Set percentile value if using the percentile statistic
   - Select output location
   - Click "Click to Calculate Statistics and Save File" to process
  
## **Interactive GUI Controls**

- Preview Panel:
  - Displays raster data with bounding box overlays
  - Automatically scales for large rasters
- Statistics Selection:
  - Checkboxes for each available statistic
  - Percentile value input (defaults to 90th percentile)
- Progress Tracking:
  - Progress bar shows calculation status
  - Status messages provide feedback

## **Common Issues and Solutions**

- Preview Display Issues:
   - Ensure both raster and CSV files are loaded
   - Verify CRS compatibility between the raster and the bounding boxes
   - For large rasters, the preview is automatically downsampled
- Statistics Calculation Errors:
   - Check that at least one statistic is selected
   - Ensure percentile values are between 0 and 100
   - Verify enough memory is available for large datasets
- Output File Issues:
   - Ensure you have write permissions for the selected directory
   - Verify the file is not open in another application

## **License**

This project is licensed under the MIT License. For details, see the LICENSE file.
