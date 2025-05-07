# **Spatial Statistics Extractor Tool**

A Python utility for extracting spatial statistics from raster data using CSV bounding boxes with transformation parameters. This tool provides an interactive interface for extracting raster data within specified regions and calculating various statistical metrics.

### **Note**

This tool is essential for analyzing spatial data in precision agriculture, environmental monitoring, and remote sensing workflows. It efficiently extracts statistical information from defined regions of interest in geospatial raster data, supporting better decision-making and data analysis.

## Table of Contents
- [**Spatial Statistics Extractor Tool**](#spatial-statistics-extractor-tool)
    - [**Note**](#note)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [1. Clone the repository:](#1-clone-the-repository)
    - [2. Install dependencies:](#2-install-dependencies)
    - [3. Run the script:](#3-run-the-script)
  - [**Features**](#features)
  - [**Requirements**](#requirements)
  - [**Input Requirements**](#input-requirements)
    - [1. Required Files](#1-required-files)
    - [2. Configuration Setup](#2-configuration-setup)
  - [**Outputs**](#outputs)
  - [**Usage Instructions**](#usage-instructions)
  - [**Interactive GUI Controls**](#interactive-gui-controls)
  - [**Common Issues and Solutions**](#common-issues-and-solutions)
  - [**License**](#license)

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

- Interactive GUI: User interface for statistical extraction
- Raster Visualization: Preview raster data and bounding boxes
- Multiple Statistics: Calculate min, max, mean, median, standard deviation, sum, count, and percentiles
- Coordinate System Support: Handle different coordinate reference systems
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
     - Column 10: Additional information
   - Example of CSV bounds file structure (8568 rows × 11 columns):
     ```
     column0,column1,column2,column3,column4,column5,column6,column7,column8,column9,column10
     1.234,5.678,0.000,0.000,-1.234,5678.901,0.000,0.000,0.000,"(0, 0), (100, 100)",additional_info
     ...
     ```

### 2. Configuration Setup
- CRS Configuration:
   - Optional EPSG code input
   - Automatic CRS detection from raster if not specified
- Statistics Selection:
   - Choose which statistics to calculate 
   - Customizable percentile values

## **Outputs**

- CSV File:
   - Contains calculated statistics for each bounding box
   - Column order: box_id, ul_x, ul_y, lr_x, lr_y, followed by statistical values
   - Example output format:
     ```
     box_id,ul_x,ul_y,lr_x,lr_y,min,max,mean,median,std,sum,count,percentile_90
     1,123456.78,7654321.98,123556.78,7654221.98,10.5,35.2,22.4,21.8,7.2,4480.5,200,32.1
     2,123556.78,7654321.98,123656.78,7654221.98,12.3,40.1,25.7,24.9,8.1,5140.5,200,36.5
     ...
     ```

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
   - Verify CRS compatibility between raster and bounds
   - For large rasters, preview is automatically downsampled
- Statistics Calculation Errors:
   - Check that at least one statistic is selected
   - Ensure percentile values are between 0 and 100
   - Verify enough memory is available for large datasets
- Output File Issues:
   - Ensure you have write permissions for the selected directory
   - Verify the file is not open in another application

## **License**

This project is licensed under the MIT License. For details, see the LICENSE file.