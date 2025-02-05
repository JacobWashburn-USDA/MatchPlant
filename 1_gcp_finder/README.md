# **GCP (Ground Control Point) Finder Tool**

A Python utility for identifying and marking Ground Control Points (GCPs) in drone imagery. This tool provides an interactive interface for matching GCP coordinates with their corresponding locations in drone-captured images.

### **Note**
This tool is handy for drone imagery processing workflows where the orthomosaic software, [OpenDroneMap open-source software](https://github.com/OpenDroneMap/ODM/tree/master), needs the GCP file to increase the precision of the orthomosaic images.

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/Ortho_to_image.git
cd Ortho_to_image/1_gcp_finder
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

### **Required Files**
1. Image Files:
   - Format: JPG/JPEG/TIF/TIFF
   - Must contain EXIF metadata with GPS coordinates
   - Located in a single directory

2. GCP File (CSV format):
   - Required columns:
     - X: Longitude/Easting
     - Y: Latitude/Northing
     - Z: Elevation
     - id: GCP identifier
    
GCP File format:
```
X,Y,Z,id
-123.456,45.789,100.5,1
-123.457,45.790,102.3,2
```

### Configuration Setup
1. Sensor Dimensions: For the information, please click [here](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/1_gcp_finder/camera_sensor_dimension.md)
   - Camera sensor width (millimeter)
   - Camera sensor height (millimeter)
     
2. Distance Threshold Configuration (meter, default = 10 m):
   - The distance threshold determines how far (in meters) from a GCP the software will search for relevant images
   - A larger threshold (e.g., 20m) will find more potential images but may include ones where the GCP is less visible
   - A smaller threshold (e.g., 5m) will be more selective, showing only images taken closer to the GCP
   - Choose based on your flight pattern:
      - For flights with dense coverage, use smaller values (5-10m)
      - For sparser coverage or higher altitude flights, use larger values (15-20m)
      - Maximum allowed value is 50m
   - The threshold helps filter out images where the GCP would be too distant or poorly visible

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
   python gcp_finder_win.py   # or gcp_finder_mac.py for macOS
   ```

2. Initial Setup:
   - Select an image folder
   - Choose GCP data file
   - Enter camera sensor dimensions: Check sensor dimension 
   - Set distance threshold 
   - Click "Click to Start Finding" to begin
  
![image](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/1_gcp_finder/images/img_1.png?raw=true)

Figure 1. Example of ininitial_setup_window

3. GCP Marking:
   - Select GCP ID from the radio buttons
   - Choose a relevant image from the list
   - Right-click to mark the GCP location
   - Press 'S' to save points
   - Press 'D' to delete last point
  
![image](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/1_gcp_finder/images/GCP_marking_window.jpg?raw=true)

Figure 2. Example of gcp_marking_window

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

This project is licensed under the MIT License. For details, see the [LICENSE](LICENSE) file.
