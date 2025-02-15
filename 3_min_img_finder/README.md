# **Minimum Image Finder Tool**

This is a Python utility for creating optimal drone image datasets for object detection models. It helps reduce dataset redundancy by selecting the minimum number of images needed to cover a target area, making object detection training more efficient.

### **Note**

This tool is handy for deep learning workflows that require creating efficient drone imagery training datasets. It helps eliminate redundant images while ensuring complete coverage of the area of interest, making object detection training more efficient and reducing computational requirements.

## Table of Contents
- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Input Requirements](#input-requirements)
- [Outputs](#outputs)
- [Usage Instructions](#usage-instructions)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [License](#license)

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/Ortho_to_image.git
cd Ortho_to_image/3_min_img_finder
```

### 2. Install dependencies:

- For Windows:
  ```bash
  pip install -r requirements_win.txt
  ```

- For macOS:
  ```bash
  pip install -r requirements_mac.txt
  ```

### 3. Run the script:

- For Windows:
  ```bash
  python min_img_finder_win.py
  ```

- For macOS:
  ```bash
  python min_img_finder_mac.py
  ```

## **Features**

- Dataset Optimization: Minimize redundancy in your training data
- Coverage Guarantee: Ensure all areas are represented in the dataset
- Smart Selection: Automatically identify unique viewpoints
- Overlap Control: Customize image overlap to balance redundancy and coverage
- Result Management: Export optimized dataset with coverage visualization

## **Requirements**

- Python 3.9+
- Dependencies:
  - NumPy
  - Pandas
  - Rasterio
  - Matplotlib
  - Shapely
  - Tkinter

## **Input Requirements**

### 1. Required Files:
- Orthophoto path:
   - Format: GeoTIFF (.tif)
   - Used as a reference for coverage analysis
- Image Folders path:
   - Orthorectified images (GeoTIFF format) - for spatial analysis
   - Original undistorted images for the final dataset
- Input Structure: All input is from the [2_odm_runner](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/2_odm_runner)
  ```
  project_root/
  ├── images/                     
  ├── odm_dem/                   
  ├── odm_orthophoto/
  │   └── odm_orthophoto.tif    # Orthophoto image
  ├── opensfm/                   
  │   └── undistorted/         
  │       └── images/           # Undistorted image folder
  └── orthorectified/           # Orthorectified image folder
  ```

### 2. Configuration Setup:
   
- Optimization Parameters: For more details, please click [here](https://github.com/JacobWashburn-USDA/MatchPlant/blob/main/3_min_img_finder/configuration_parameters_guide.md)
   - Flight line width (%) - controls spacing between selected images
   - Horizontal minimum/maximum overlap (%) - ensures adequate coverage
   - Vertical minimum/maximum overlap (%) - manages redundancy between flight lines
   - Uncovered area threshold (%) - sets acceptable coverage gaps

## **Outputs**

- Optimized Dataset:
   - Selected_undistorted_images folder - Minimum number of images
   - Selected_images_list file - Coverage report (.csv)
   - Coverage_plot image - Visual coverage map (png)
- Output structure:
  ```
  output_folder/
  ├── selected_undistorted_images_YYYYMMDD_HHMMSS/  # Your optimized dataset
  │   └── selected_images...
  ├── selected_images_list_YYYYMMDD_HHMMSS.csv      # Selection metadata
  └── coverage_plot_YYYYMMDD_HHMMSS.png             # Coverage visualization
  ```

## **Usage Instructions**

- Launch Application:
   ```python
   python min_img_finder_win.py  # or min_img_finder_mac.py for macOS
   ```
- Initial Setup:
   - Select your reference orthophoto
   - Choose your input image folders
   - Configure optimization parameters based on your needs
   - Click "Click to Start Finding" to begin optimization
  
![image](https://github.com/JacobWashburn-USDA/MatchPlant/blob/main/3_min_img_finder/images/img_1.png?raw=true)

Figure 1. Example of ininitial_setup_window
- Dataset Creation:
   - Review coverage to ensure all areas of interest are included
   - Save your optimized dataset
  
![image](https://github.com/JacobWashburn-USDA/MatchPlant/blob/main/3_min_img_finder/images/img_2.png?raw=true)

Figure 2. Example of min_img_window

## **Common Issues and Solutions**

- Missing Coverage:
   - Decrease the flight line width
   - Increase overlap parameters
   - Lower the uncovered area threshold
- Too Many Images:
   - Increase flight line width
   - Decrease maximum overlap
   - Increase uncovered area threshold
- Processing Issues:
   - Ensure consistent image georeferencing
   - Verify image format compatibility
   - Check available system memory

## **License**

This project is licensed under the MIT License. For details, see the [LICENSE](https://github.com/JacobWashburn-USDA/MatchPlant/blob/main/LICENSE) file.
