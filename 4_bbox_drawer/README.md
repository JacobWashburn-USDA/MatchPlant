# **Bounding Box Annotation Tool**

This Python utility creates annotated image datasets for object detection models. It provides an interactive interface for drawing bounding boxes around objects of interest and supports COCO (.json file) and YOLO (.txt file) annotation formats with Windows and macOS platform implementations.

### **Note**
This tool is essential for deep learning workflows that require creating annotated training datasets. It provides an intuitive interface for drawing bounding boxes, managing multiple categories, and exporting annotations in standard formats, making the dataset creation process more efficient and organized.

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/JacobWashburn-USDA/Ortho_to_image.git
cd Ortho_to_image/4_bbox_drawer
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
python bbox_drawer_win.py
```

For macOS:
```bash
python bbox_drawer_mac.py
```

## **Features**

- Multi-Category Support: Define up to 5 different annotation categories
- Format Flexibility: Export annotations in COCO or YOLO format
- Interactive Drawing: Real-time visualization of bounding boxes
- Category Management: Quick category switching with color coding
- Mask Generation: Automatic creation of masked images
- Progress Tracking: Save and resume annotation sessions

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
- Image folder path:
  - Click [here](https://github.com/JacobWashburn-USDA/Ortho_to_image/tree/main/3_min_img_finder) / The image input folder is created by "3_min_img_finder" method - the "selected_undistorted_images" folder
  - Supported formats:
    - JPG/JPEG
    - PNG
    - TIF/TIFF

Input structure:


```
project_root/
├── images/                     # Your input image folder
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

### Configuration Setup
1. Initial Configuration:
   - Annotation format selection (COCO/YOLO): For more details, please click [here](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/4_bbox_drawer/annotation_format.md)
     - COCO - JSON file format
     - YOLO - TXT file format

2. Annotation Parameters:
   - Category definition (1-5 categories, default = 1) 
   - Category names (default = maize)

## **Outputs**

1. Annotation Results:
   - Annotated images with bounding boxes
   - Masked images for each annotation
   - Annotation files in the chosen format

Output structure:
```
project_root/
├── mask_box/                   # Individual masked images and annotations
│   ├── image1_1.jpg
│   ├── image1_1.json/txt
│   └── ...
└── img_box/                    # Images with visualized annotations and annotations
    ├── image1.jpg
    ├── image1_annotations.json/txt
    └── ...
```

## **Usage Instructions**

1. Launch Application:
   ```python
   python bbox_drawer_win.py  # or bbox_drawer_mac.py for macOS
   ```

2. Initial Setup:
   - Select your image folder
   - Choose annotation format (COCO/YOLO)
   - Define categories
   - Click "Click to Start Drawing" to begin annotation
  
![image](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/4_bbox_drawer/images/img_1.png?raw=true)

Figure 1. Example of ininitial_setup_window

3. Drawing Boxes:
   - Right-click and drag to draw boxes
   - Use number keys (1-5) to switch categories
   - Press 'D' to delete the last box
   - Press 'Enter' to move to the next image
   - Press 'ESC' to exit
  
![image](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/4_bbox_drawer/images/img_2.png?raw=true)

Figure 2. Example of bbox_drawer_window

## **Annotation Strategy**

The tool supports efficient annotation by:
1. Providing real-time visual feedback
2. Enabling quick category switching
3. Supporting multiple annotation formats
4. Automating mask generation
5. Maintaining an organized output structure

## **Common Issues and Solutions**

1. Drawing Issues:
   - Use right-click for drawing
   - Ensure proper window focus
   - Check the mouse cursor position

2. Category Management:
   - Use number keys for switching
   - Verify category selection
   - Check category color coding

3. File Management:
   - Verify image format compatibility
   - Check write permissions
   - Monitor disk space

## **License**

This project is licensed under the MIT License. For details, see the [LICENSE](LICENSE) file.
