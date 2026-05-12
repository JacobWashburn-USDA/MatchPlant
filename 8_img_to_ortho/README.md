# **Object Detection Projection Tool**

A Python utility for projecting detected bounding boxes from undistorted UAV images onto orthomosaics using forward projection. This module bridges object detection results (Module 7) with spatial analysis (Module 9) by transforming pixel coordinates into georeferenced coordinates.

### **Note**

This tool is designed for the MatchPlant pipeline. It takes predictions from trained object detection models and projects them onto orthomosaics using camera parameters and terrain elevation data from OpenDroneMap. The output provides spatially accurate bounding box locations for downstream phenotypic analysis.

## Table of Contents
- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Input & Output](#input--output)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Quick Start

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run from Module 7 test output:
```bash
python project_boxes.py \
  --dataset-path /path/to/odm_project \
  --predictions-json ../6-1_obj_det_trainer/runs/my_training_run/test_results/coco_predictions.json \
  --annotation-json ../annotations/test.json \
  --output-dir orthorectified2 \
  --num-threads 4
```

`--annotation-json` is important when Module 7 saves predictions as a flat COCO prediction list, because Module 8 needs the test `image_id -> file_name` mapping.

### 3. Alternative: configure paths in the script:
You can still edit `project_boxes.py` directly if you prefer running from an IDE:
```python
dataset_path = '/path/to/your/odm_project'
predictions_json_path = "test_results/coco_predictions.json"
annotation_json_path = "annotations/test.json"
tile_metadata_path = None
```

### 4. Get results:
Output saved to: `dataset_path/orthorectified2/projected_boxes.csv`

---

## Features

- **Forward Projection**: Projects bounding boxes using camera parameters and DSM
- **Tile Support**: Handles coordinate transformation from tiled test images
- **Module 9 Compatible**: Outputs CSV format directly usable in spatial statistics extraction
- **Multiprocessing**: Parallel processing for faster computation
- **Geospatial Output**: Produces georeferenced coordinates for GIS workflows

---

## Requirements

- Python 3.9+
- Dependencies (see `requirements.txt`):
  - rasterio>=1.3.0 (DEM/DSM handling)
  - numpy>=1.24.0
  - pandas>=2.0.0
  - opencv-python>=4.8.0
  - scikit-image>=0.21.0

---

## Input & Output

### **Inputs Required:**

1. **From Module 7** - `coco_predictions.json` or full COCO prediction JSON
   
   Module 8 supports both common Module 7 output styles:
   - A flat COCO prediction list, for example `[ {"image_id": 123, "bbox": [x, y, w, h], "score": 0.89} ]`
   - A full COCO-style dictionary containing `images` and `annotations`

   If using the flat list, also provide the test annotation file with `--annotation-json`.

2. **From Module 2 (ODM)** - Required files:
   - `opensfm/undistorted/images/*.JPG` - Undistorted images
   - `opensfm/undistorted/reconstruction.json` - Camera parameters
   - `odm_dem/dsm.tif` - Digital Surface Model
   - `odm_georeferencing/coords.txt` - Coordinate offsets

3. **From Module 5** (Optional) - `tile_metadata.json`
   - Recommended if Module 7 tested on tiled images
   - Format: `{"tile.jpg": {"original_image": "DJI_0123.JPG", "offset_x": 0, "offset_y": 0}}`
   - If metadata is not provided, Module 8 attempts to infer offsets from row/column tile names such as `0688_548.JPG_r2c3.tif`.

### **Output:**

**File**: `projected_boxes.csv` (13 columns, no headers)

| Columns | Content | Example |
|---------|---------|---------|
| 1-6 | Affine transform (a,b,c,d,e,f) | 0.003, 0.0, 562340.234, ... |
| 7-9 | Placeholders (zeros) | 0.0, 0.0, 0.0 |
| 10 | Dimensions "(0,0),(w,h)" | "(0, 0), (85, 120)" |
| 11 | Filename | DJI_0123_box_1.tif |
| 12| Detection ID | 1 |
| 13 | Confidence score | 0.92 |

**Example row:**
```csv
0.003,0.0,562340.234,0.0,-0.003,4512890.567,0.0,0.0,0.0,"(0, 0), (85, 120)",DJI_0123_box_1.tif,1,0.92
```

---

## Usage

### **Basic Workflow:**

1. **Complete Prerequisites:**
   - Module 2: ODM processing finished
   - Module 7: Object detection testing complete with `predictions.json`

2. **Run Script:**
   ```bash
   python project_boxes.py \
     --dataset-path /path/to/odm_project \
     --predictions-json /path/to/test_results/coco_predictions.json \
     --annotation-json /path/to/annotations/test.json \
     --output-dir orthorectified2 \
     --num-threads 4
   ```

   You can also open `project_boxes.py` in your IDE and edit the default paths at the top.

4. **Verify Output:**
   ```bash
   # Check file exists
   ls dataset_path/orthorectified2/projected_boxes.csv
   
   # Check format (should have 13 columns, no headers)
   head -1 projected_boxes.csv
   ```

5. **Use in Module 9:**
   - Open Module 9 GUI
   - Load `projected_boxes.csv` as bounds file
   - Load your raster (CHM, NDVI, etc.)
   - Extract spatial statistics

---

## Configuration

All settings are in `project_boxes.py` (lines 30-48):

### **Command-line settings:**
```bash
python project_boxes.py \
  --dataset-path /path/to/odm_project \
  --predictions-json /path/to/test_results/coco_predictions.json \
  --annotation-json /path/to/annotations/test.json \
  --tile-metadata /path/to/tile_metadata.json \
  --output-dir orthorectified2 \
  --num-threads 4
```

All command-line settings are optional if you prefer editing the defaults in `project_boxes.py`.

Useful options:
```bash
--dem-filename odm_dem/dsm.tif
--interpolation bilinear
--visibility-test
```

### **Performance Tuning:**

**Speed:**
- Increase `num_threads` (uses all cores by default)
- Keep `skip_visibility_test = True`

**Accuracy:**
- Set `skip_visibility_test = False` (slower but removes artifacts)
- Use high-resolution DSM

**Memory:**
- Reduce `num_threads` if running out of memory
- Set `num_threads = 1` for debugging

---

## Troubleshooting

### **Common Issues:**

**"Image not found in reconstruction"**
- Check that tile names map to undistorted image names
- Verify tile metadata if you used tiled images
- Check files in `opensfm/undistorted/images/`

**"No valid pixels found"**
- Bounding box is outside DSM coverage
- Check DSM covers the entire area
- Verify tile offsets are correct

**"Multithreading error"**
- Reduce `num_threads` in configuration
- Try `num_threads = 1` for debugging
- Check available system memory

**Boxes do not align in Module 9**
- Missing or incorrect tile metadata
- Re-run Module 7 on original images (not tiles)
- Check the coordinate system of DSM

**"Cannot find coords.txt"**
- Re-run Module 2 with the proper GCP file
- Check `odm_georeferencing/coords.txt` exists

### **Validation:**

```python
# Check predictions.json
import json
with open('test_results/predictions.json') as f:
    data = json.load(f)
    print(f"Detections: {len(data['annotations'])}")

# Check ODM files
import os
assert os.path.exists('odm_project/odm_dem/dsm.tif'), "DSM missing"
assert os.path.exists('odm_project/opensfm/undistorted/reconstruction.json'), "Reconstruction missing"
print("All files found")

# Check output format
import pandas as pd
df = pd.read_csv('orthorectified2/projected_boxes.csv', header=None)
print(f"Columns: {len(df.columns)}")  # Should be 13
print(f"Rows: {len(df)}")             # Should match number of detections
```

---

## License

This project is licensed under the MIT License. For details, see the [LICENSE](https://github.com/JacobWashburn-USDA/MatchPlant/blob/main/LICENSE) file.

---

**Pipeline Flow:**
```
Module 7 (predictions.json) → Module 8 (projected_boxes.csv) → Module 9 (trait extraction)
```

**Related Modules:**
- ← [Module 7: Object Detection Testing](../7_obj_det_tester/README.md)
- → [Module 9: Spatial Statistics Extraction](../9_spatial_stats_extractor/README.md)
