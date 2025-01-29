# **Bounding Box Annotation Tool**

A Python utility for creating annotated image datasets for object detection models. This tool provides an interactive interface for drawing bounding boxes around objects of interest, supporting both COCO and YOLO annotation formats, with implementations for both Windows and macOS platforms.

### **Note**
This tool is essential for deep learning workflows where you need to create annotated training datasets. It provides an intuitive interface for drawing bounding boxes, managing multiple categories, and exporting annotations in standard formats, making the dataset creation process more efficient and organized.

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/YourUsername/BoundingBoxAnnotator.git
cd BoundingBoxAnnotator
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

### Configuration Setup
1. Initial Configuration:
   - Image directory selection
   - Annotation format selection (COCO/YOLO)
   - Category definition (1-5 categories)
   
2. Annotation Parameters:
   - Category names
   - Output format preferences
   - File naming conventions

### **Required Files**
- Image folder containing supported formats:
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

## **Outputs**

1. Annotation Results:
   - Annotated images with bounding boxes
   - Masked images for each annotation
   - Annotation files in chosen format

Output structure:
```
project_root/
├── mask_box/                   # Individual masked images and annotations
│   ├── image1_1.jpg
│   ├── image1_1.json/txt
│   └── ...
└── img_box/                    # Images with visualized annotations
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
   - Click "Start Drawing" to begin annotation

3. Drawing Boxes:
   - Right-click and drag to draw boxes
   - Use number keys (1-5) to switch categories
   - Press 'D' to delete last box
   - Press 'Enter' to move to next image
   - Press 'ESC' to exit

## **Annotation Strategy**

The tool supports efficient annotation by:
1. Providing real-time visual feedback
2. Enabling quick category switching
3. Supporting multiple annotation formats
4. Automating mask generation
5. Maintaining organized output structure

## **Common Issues and Solutions**

1. Drawing Issues:
   - Use right-click for drawing
   - Ensure proper window focus
   - Check mouse cursor position

2. Category Management:
   - Use number keys for switching
   - Verify category selection
   - Check category color coding

3. File Management:
   - Verify image format compatibility
   - Check write permissions
   - Monitor disk space

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.