# **Image Splitter Tool**

This Python utility splits large images and their corresponding COCO format annotations into smaller tiles for training object detection models. It provides a graphical interface for configuring splitting parameters and supports Windows and macOS platforms.

### **Note**
This tool is essential for preparing large-scale imagery datasets for deep-learning model training. It handles the complexities of splitting images and their annotations while maintaining proper object relationships and coordinates. The tool is handy for aerial photography and other large-format images.

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/Ortho_to_image.git
cd Ortho_to_image/5_img_splitter
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
python img_splitter_win.py
```

For macOS:
```bash
python img_splitter_mac.py
```

## **Features**

- Configurable Tile Size: Define target tile dimensions
- Overlap Control: Set pixel overlap between adjacent tiles
- Dataset Splitting: Automatic train/validation/test set creation
- COCO Format Support: Maintains COCO annotation format
- Interactive Interface: GUI for parameter configuration
- Progress Tracking: Visual feedback during processing
- Results Summary: Detailed statistics after processing

## **Requirements**

- Python 3.x
- Dependencies:
  - OpenCV (cv2)
  - NumPy
  - Matplotlib
  - Tkinter (Windows)
  - PyObjC (macOS)

## **Input Requirements**

### **Required Files**
1. Image folder path
  - Click [here](https://github.com/JacobWashburn-USDA/Ortho_to_image/tree/main/3_min_img_finder) / The image input folder is created by "3_min_img_finder" method - the "selected_undistorted_images" folder
  - supported formats:
    - TIF/TIFF
    - JPG/JPEG
    - PNG
2. Annotation folder path
  - Click [here](https://github.com/JacobWashburn-USDA/Ortho_to_image/tree/main/4_bbox_drawer) / The annotation input folder is created by "4_bbox_drawer" method - the "img_box" folder
  - Annotation file - COCO format in JSON files

Input structure:
```
project_root/
├── images/                     # Your input image folder
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
└── annotations/               # Your input annotation folder
    ├── image1_annotations.json
    ├── image2_annotations.json
    └── ...
```
3. Output folder path
  
### Configuration Setup
1. Dataset Split Configuration:
   - Train split ratio (default: 0.7)
   - Validation split ratio (default: 0.15)
   - Test split ratio (default: 0.15)
   
   *Dataset split ratios must sum to 1.0 (100% of the data)

2. Tile Configuration:
   - Target tile size (default: 1024 pixels)
      - The desired dimension for output tiles
      - Larger images are divided into a grid of this approximate size
      - If an image is smaller than the target size, it remains unchanged
      - Recommended range: 512-2048 pixels, depending on your model's requirements
   
   - Overlap pixels (default: 100 pixels)
      - Number of pixels that adjacent tiles overlap
      - Helps prevent objects from being cut off at tile boundaries
      - Larger overlap helps with object detection at tile edges
      - Recommended range: 10-20% of target tile size
   
   - Minimum IoU threshold (default: 0.3)
      - Minimum Intersection over Union required to keep a split object
      - Controls how much of an object must be within a tile to be included
      - Higher values (e.g., 0.5) ensure more complete objects
      - Lower values (e.g., 0.1) retain more partial objects
      - Range: 0.0-1.0 (0 keeps all intersections, 1 requires complete objects)
   
   - Minimum object size (default: 10 pixels)
      - Minimum size (width or height) required for split objects
      - Prevents tiny object fragments from being included
      - Objects smaller than this are filtered out after splitting
      - Recommended range: 5-20 pixels, depending on your model's requirements

## **Outputs**

1. Split Results:
   - Tiled images organized by dataset split
   - Updated COCO annotations for each split
   - Processing statistics

Output structure:
```
output_dir/
├── data/                      # Split image tiles
│   ├── train/
│   │   ├── image1_r1c1.tif
│   │   └── ...
│   ├── val/
│   │   ├── image1_r1c1.tif
│   │   └── ...
│   └── test/
│       ├── image1_r1c1.tif
│       └── ...
└── annotations/              # Split annotations
    ├── train.json
    ├── val.json
    └── test.json
```

## **Usage Instructions**

1. Launch Application:
   ```python
   python img_splitter_win.py  # or img_splitter_mac.py for macOS
   ```

2. Initial Setup:
   - Select an input image folder
   - Select an input annotation folder
   - Choose output directory
   - Configure split ratios
   - Set tile parameters
   - Click "Click to Start Processing" to begin
     
![image](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/5_img_splitter/images/img1.png?raw=true)

Figure 1. Example of ininitial_setup_window

3. Processing Result:
   - Wait for processing to complete
   - Review results summary
  
![image](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/5_img_splitter/images/img2.png?raw=true)

Figure 2. Example of processing_result_window

## **Processing Strategy**

The tool processes images and annotations by:
1. Calculating optimal tile sizes
2. Adding configurable overlap
3. Adjusting bounding box coordinates
4. Maintaining annotation relationships
5. Organizing split datasets

## **Common Issues and Solutions**

1. Memory Issues:
   - Reduce tile size for large images
   - Process fewer images at once
   - Close other applications

2. Annotation Handling:
   - Verify COCO format compliance
   - Check annotation-image correspondence
   - Monitor IoU thresholds

3. File Management:
   - Verify image format compatibility
   - Check write permissions
   - Monitor disk space

## **License**

This project is licensed under the MIT License. For details, see the [LICENSE](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/LICENSE) file.
