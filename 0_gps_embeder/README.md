## **GPS Embedding Tool for Drone Imagery**

A Python utility for processing drone imagery and embedding GPS coordinates by matching image timestamps with event data. This tool handles UTM (Universal Transverse Mercator) to WGS84 coordinate conversion and EXIF metadata manipulation.

### **Features**

- Timestamp Matching: Automatically matches drone images with GPS coordinates based on timestamp data
- Coordinate Conversion: Converts UTM coordinates to WGS84 (latitude/longitude) format
- EXIF Processing: Reads and writes GPS data to image EXIF metadata
- Error Analysis: Provides detailed timing error analysis for matched images
- Batch Processing: Handles multiple images in a single operation
- Progress Tracking: Includes progress monitoring for large batches of images

### **Processing Pipeline**

- **Setup**: Initializes the tool and sets UTM zone parameters
- **Input Processing**: Loads and validates image files and event data
- **Time Matching**: Matches images with events based on timestamp correlation
- **Coordinate Processing**: Converts UTM coordinates to GPS format
- **Output Generation**: Generates detailed reports and embeds GPS data in images

### **Requirements**

- Python 3.x
- Dependencies:
  - piexif: For EXIF metadata manipulation
  - PIL (Pillow): For image processing
  - pyproj: For coordinate system transformations
  - Other standard Python libraries (csv, os, shutil, datetime)

### **Input Requirements**

- **User Inputs**
  - UTM Zone Information:
    - Zone number (1-60)
    - Hemisphere (N/S)
    - Example: Zone 15N for Columbia, MO United States
  - Folder Path:
    - Base folder containing:
      - Drone images (.jpg/.jpeg)
      - events.txt file with coordinate data
- **File Requirements**
  project
  |-- images/
      |--image_1.jpg
      |--image_2.jpg
  |-- events.txt

  - Image Files:
    - Format: JPG/JPEG
    - Must contain EXIF timestamp data
    - Should be named in a way that maintains chronological order
  - Events File (events.txt):
    - Must be named exactly “events.txt”
    - Tab-delimited format
    - Required columns:
      - Timestamp (in seconds)
      - X (UTM Easting)
      - Y (UTM Northing)
      - Z (Altitude)
      ![Example of events.txt](https://raw.githubusercontent.com/JacobWashburn-USDA/Ortho_to_image/main/0_gps_embeder/images/img_events_txt.png)
      - File must be placed in the same folder as images

### **Outputs**

- Program automatically creates:
  - “events_with_matches.csv”: Contains matching results
    - CSV file with matched events and timing analysis
  - “images_with_gps” folder: Contains processed images
    - Copy of original images with embedded GPS coordinates

### **Interactive Usage**

- Run the code
- Follow the interactive prompts:
  - Enter UTM zone number (1-60): 15
  - Enter hemisphere (N/S): N
  - Enter the path to your images folder: /path/to/your/drone/images
- The code will:
  - Search for images in the specified folder
  - Read the events.txt file
  - Process and match timestamps
  - Create GPS-tagged copies of images
  - Generate a matching report

### **Note**

This tool is particularly useful for drone imagery processing workflows where GPS data needs to be embedded post-flight or when working with systems that record position data separately from image capture.