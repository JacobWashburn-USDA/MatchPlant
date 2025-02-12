# **GPS Embedding Tool for Drone Imagery**

A Python utility for processing drone imagery and embedding GPS coordinates by matching image timestamps with event data. This tool handles UTM (Universal Transverse Mercator) to WGS84 coordinate conversion and EXIF metadata manipulation.

### **Note**
This tool is handy for drone imagery processing workflows where GPS data must be embedded post-flight or when working with systems that record position data separately from image capture.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/Ortho_to_image.git
cd Ortho_to_image/0_gps_embeder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python 0_gps_embed.py
```

## **Features**

- Timestamp Matching: Automatically matches drone images with GPS coordinates based on timestamp data
- Coordinate Conversion: Converts UTM coordinates to WGS84 (latitude/longitude) format
- EXIF Processing: Reads and writes GPS data to image EXIF metadata
- Error Analysis: Provides detailed timing error analysis for matched images
- Batch Processing: Handles multiple images in a single operation
- Progress Tracking: Includes progress monitoring for large batches of images

## **Requirements**

- Python 3.x
- Dependencies:
  - piexif: For EXIF metadata manipulation
  - PIL (Pillow): For image processing
  - pyproj: For coordinate system transformations
  - Other standard Python libraries (csv, os, shutil, datetime)

## **Input Requirements**

- ### **User Inputs**
1. UTM Zone Information:
    - Zone number (1-60)
    - Hemisphere (N/S)
    - Example: Zone 15N for Columbia, MO, United States
2. Folder Path:
    - Base folder containing:
      - Drone images (.jpg/.jpeg)
      - events.txt file with coordinate data
- ### **File Requirements**
1. Image Files:
    - Format: JPG/JPEG
    - Must contain EXIF timestamp data
    - Should be named in a way that maintains chronological order
2. Events File (events.txt):
    - Must be named exactly “events.txt”
    - Tab-delimited format
    - Required columns:
      - Timestamp (in seconds)
      - X (UTM Easting)
      - Y (UTM Northing)
      - Z (Altitude)
      - File must be placed in the same folder as images
     
![image](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/0_gps_embeder/images/img_events_txt.png?raw=true)

Figure 1. Example of events.txt file
      
## **Outputs**

- Program automatically creates:
  - “events_with_matches.csv”: Contains matching results
    - CSV file with matched events and timing analysis
  - “images_with_gps” folder: Contains processed images
    - Copy of original images with embedded GPS coordinates
```
your_input_folder/
├── events.txt
├── your_images.jpg
├── events_with_matches.csv   # Generated matching results
└── images_with_gps/          # Generated folder with GPS-tagged images
    └── your_images.jpg       # Copies of images with GPS data
```

## **Usage Examples**

### **Interactive Usage**

1. Run the script:
   ```bash
   python 0_gps_embed.py
   ```

2. Follow the interactive prompts:
   ```bash
   Enter UTM zone number (1-60): 15
   Enter hemisphere (N/S): N
   Enter the path to your images folder: /path/to/your/drone/images
   ```

### **Programmatic Usage**

```python
from gps_embed import TimestampMatcher

# Initialize the matcher
matcher = TimestampMatcher()

# Set UTM zone
matcher.set_utm_zone(15, 'N')

# Process images
image_files = matcher.get_image_files('path/to/images')
matcher.load_events_file('path/to/events.txt')
matcher.find_initial_offset(image_files[0])

# Generate output
matcher.save_matched_events('events.txt', 'events_with_matches.csv')
matcher.embed_gps_to_images('input_folder', 'images_with_gps')
```

## **Common Issues and Solutions**

1. **No EXIF Data**:
   - Ensure images are not copies or screenshots
   - Verify the camera is saving EXIF metadata

2. **Timestamp Matching Errors**:
   - Check camera clock synchronization
   - Verify events.txt timestamps are in the correct format
   - Ensure images are in chronological order

3. **UTM Conversion Issues**:
   - Confirm the correct UTM zone for your location
   - Verify coordinate format in events.txt
     
## **License**

This project is licensed under the MIT License. For details, see the [LICENSE](LICENSE) file.
