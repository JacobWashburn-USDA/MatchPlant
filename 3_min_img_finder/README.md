# **Minimum Image Finder**

A Python utility for finding the minimum set of drone images required to cover a target area while maintaining specified overlap requirements. This tool provides an interactive interface for optimizing image selection and reducing processing time in drone imagery workflows.

### **Note**
This tool is particularly useful for drone imagery processing workflows where you need to minimize the number of images while ensuring adequate coverage, especially when using photogrammetry software like OpenDroneMap, Pix4D, or Agisoft Metashape.

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/YourUsername/MinimumImageFinder.git
cd MinimumImageFinder
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

- Interactive GUI: User-friendly interface for image optimization
- Smart Selection: Automatically identifies optimal image sequences
- Coverage Analysis: Visual representation of covered and uncovered areas
- Overlap Control: Customizable horizontal and vertical overlap requirements
- Result Management: Export selected images and detailed reports
- Progress Tracking: Visual display of selected image sequences
- Flight Line Analysis: Automatic detection and optimization of flight lines

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
   - Orthophoto path (.tif)
   - Orthorectified image folder
   - Undistorted image folder

2. Optimization Parameters:
   - Flight line width (%)
   - Horizontal minimum/maximum overlap (%)
   - Vertical minimum/maximum overlap (%)
   - Uncovered area threshold (%)

### **Required Files**
1. Orthophoto:
   - Format: GeoTIFF (.tif)
   - Must contain proper georeferencing

2. Image Folders:
   - Orthorectified images (GeoTIFF format)
   - Undistorted original images

## **Outputs**

1. Selected Images:
   - Timestamped folder containing selected undistorted images
   - CSV file with sequence information
   - Coverage plot showing selected images and uncovered areas

Output structure:
```
output_folder/
├── selected_undistorted_images_YYYYMMDD_HHMMSS/
│   └── selected_images...
├── selected_images_list_YYYYMMDD_HHMMSS.csv
└── coverage_plot_YYYYMMDD_HHMMSS.png
```

## **Usage Instructions**

1. Launch Application:
   ```python
   python min_img_finder_win.py  # or min_img_finder_mac.py for macOS
   ```

2. Initial Setup:
   - Select orthophoto file (.tif)
   - Choose orthorectified image folder
   - Select undistorted image folder
   - Set optimization parameters
  
3. Optimization Process:
   - Click "Start Finding" to begin optimization
   - Review coverage visualization
   - Save results if satisfied

## **Interactive Controls**

The tool provides two main windows:

1. Configuration Window:
   - File/folder selection via browse buttons
   - Parameter input fields
   - Start button to begin optimization

2. Results Window:
   - Coverage visualization
   - Save results button
   - Coverage statistics display

## **Common Issues and Solutions**

1. File Loading Issues:
   - Verify file paths and permissions
   - Check image format compatibility
   - Ensure proper georeferencing in orthophoto

2. Optimization Problems:
   - Adjust overlap parameters
   - Check flight line width settings
   - Verify uncovered area threshold

3. Memory Errors:
   - Reduce image resolution if necessary
   - Process smaller areas separately
   - Close other memory-intensive applications

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgments**

- Developed by Worasit Sangjan
- Contributors welcome

## **Contributing**

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request