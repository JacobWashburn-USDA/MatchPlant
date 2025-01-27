# **Minimum Image Finder**

A Python utility for creating optimal drone image datasets for object detection models. This tool helps reduce dataset redundancy by selecting the minimum number of images needed to cover a target area, making the object detection training more efficient.

### **Note**
This tool is handy for deep learning workflows where you must create efficient drone imagery training datasets. It helps eliminate redundant images while ensuring complete coverage of your area of interest, making the object detection training more efficient and reducing computational requirements.

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/YourUsername/MinimumImageFinder.git
cd Ortho_to_image/3_min_img_finder
```

### 2. Install dependencies:

For Windows:
```bash
pip install -r requirements_win.txt
```

For macOS:
```bash
pip install -r requirements_mac.txt
```

### 3. Run the script:

For Windows:
```bash
python min_img_finder_win.py
```

For macOS:
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

- Python 3.x
- Dependencies:
  - NumPy
  - Pandas
  - Rasterio
  - Matplotlib
  - Shapely
  - Tkinter

## **Input Requirements**

### Configuration Setup
1. Path Configuration:
   - Orthophoto path (.tif) - for reference and coverage verification
   - Orthorectified image folder - for spatial analysis
   - Undistorted image folder - contains original images for the dataset

2. Optimization Parameters:
   - Flight line width (%) - controls spacing between selected images
   - Horizontal minimum/maximum overlap (%) - ensures adequate coverage
   - Vertical minimum/maximum overlap (%) - manages redundancy between flight lines
   - Uncovered area threshold (%) - sets acceptable coverage gaps

### **Required Files**
1. Orthophoto:
   - Format: GeoTIFF (.tif)
   - Used as a reference for coverage analysis

2. Image Folders:
   - Orthorectified images (GeoTIFF format)
   - Original undistorted images for the final dataset

## **Outputs**

1. Optimized Dataset:
   - Selected images with minimal redundancy
   - Coverage report (CSV)
   - Visual coverage map

Output structure:
```
output_folder/
├── selected_undistorted_images_YYYYMMDD_HHMMSS/  # Your optimized dataset
│   └── selected_images...
├── selected_images_list_YYYYMMDD_HHMMSS.csv      # Selection metadata
└── coverage_plot_YYYYMMDD_HHMMSS.png             # Coverage visualization
```

## **Usage Instructions**

1. Launch Application:
   ```python
   python min_img_finder_win.py  # or min_img_finder_mac.py for macOS
   ```

2. Initial Setup:
   - Select your reference orthophoto
   - Choose your input image folders
   - Configure optimization parameters based on your needs
  
3. Dataset Creation:
   - Click "Start Finding" to begin optimization
   - Review coverage to ensure all areas of interest are included
   - Save your optimized dataset

## **Optimization Strategy**

The tool optimizes your dataset by:
1. Analyzing flight lines to maintain systematic coverage
2. Eliminating redundant, overlapping images
3. Ensuring representation of all unique viewpoints
4. Maintaining minimum required overlap for complete coverage
5. Selecting images that maximize area coverage while minimizing count

## **Common Issues and Solutions**

1. Missing Coverage:
   - Decrease the flight line width
   - Increase overlap parameters
   - Lower the uncovered area threshold

2. Too Many Images:
   - Increase flight line width
   - Decrease maximum overlap
   - Increase uncovered area threshold

3. Processing Issues:
   - Ensure consistent image georeferencing
   - Verify image format compatibility
   - Check available system memory

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
