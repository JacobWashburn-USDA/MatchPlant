# **GCP (Ground Control Point) Finder**

A Python utility for identifying and marking Ground Control Points (GCPs) in drone imagery. This tool provides an interactive interface for matching GCP coordinates with their corresponding locations in drone-captured images.

### **Note**
This tool is handy for drone imagery processing workflows where the orthomosaic software needs the GCP file to increase the precision of orthomosaic image, for example when using OpenDroneMap open-source software.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/Ortho_to_image.git
cd Ortho_to_image/1_gcp_finder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:

For Windows:
```bash
python gcp_finder_win.py
```

For macOS:
```bash
python gcp_finder_mac.py
```

## **Features**

- Interactive GUI: User-friendly interface for GCP identification
- Image Navigation: Pan and zoom capabilities for precise point marking
- Coordinate Transformation: Automatic conversion between WGS84 and UTM coordinates
- Metadata Extraction: Comprehensive EXIF data parsing from drone imagery
- Smart Image Finding: Automatically identifies images likely to contain specific GCPs
- Point Management: Save, delete, and review marked GCP locations
- Progress Tracking: Visual display of marked points and coordinates

## **Requirements**

- Python 3.x
- Dependencies:
  - OpenCV (cv2)
  - NumPy
  - Pandas
  - PIL (Pillow)
  - Matplotlib
  - PyProj
  - Tkinter

## **Input Requirements**

### Configuration Setup
1. Sensor Dimensions:
   - Camera sensor width (mm)
   - Camera sensor height (mm)
   - Distance threshold (meters)

### **Required Files**
1. Image Files:
   - Format: JPG/JPEG/TIF/TIFF
   - Must contain EXIF metadata with GPS coordinates
   - Located in a single directory

2. GCP File (CSV format):
   - Required columns:
     - id: GCP identifier
     - X: Longitude/Easting
     - Y: Latitude/Northing
     - Z: Elevation
    
![image]()

Figure 1. Example of GCP file

## **Outputs**

1. gcp_list.txt:
   - Format: "UTM_X UTM_Y Elevation ImageX ImageY Filename GCP_ID"
   - Contains all marked GCP locations
   - Includes coordinate transformations
   - Headers specify coordinate system

GCP list format:
```
WGS84 UTM [zone][hemisphere]
[UTM_X] [UTM_Y] [Elevation] [ImageX] [ImageY] [Filename] [GCP_ID]
```

The program creates `gcp_list.txt` in your input image folder with this format:
```
input_image_folder/
├── your_images.jpg
├── gcp_list.txt        # Generated GCP file
```


## **Usage Instructions**

1. Launch Application:
   ```python
   python gcp_finder_win.py
   ```

2. Initial Setup:
   - Select an image folder
   - Choose GCP data file
   - Enter camera sensor dimensions
   - Set distance threshold

3. GCP Marking:
   - Select GCP ID from the radio buttons
   - Choose a relevant image from the list
   - Right-click to mark the GCP location
   - Press 'S' to save points
   - Press 'D' to delete last point

## **Interactive Controls**

- Mouse Controls:
  - Right Click: Mark GCP location
  - Left Click + Drag: Pan image
  - Scroll Wheel: Zoom in/out

- Keyboard Controls:
  - 'S': Save marked points
  - 'D': Delete last point

## **Common Issues and Solutions**

1. No Images Listed:
   - Verify image folder path
   - Check image EXIF data integrity
   - Confirm GCP coordinates are within range

2. Coordinate Transformation Errors:
   - Validate GCP file format
   - Check coordinate system consistency
   - Verify UTM zone information

3. Image Loading Failures:
   - Ensure supported image format
   - Check file permissions
   - Verify image file integrity

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
