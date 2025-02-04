
# **Annotation Formats**

The tool supports two popular annotation formats:

## COCO Format

Individual annotation file example (`image1_1.json`):
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080,
      "date_captured": ""
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "maize",
      "supercategory": "plant"
    },
    {
      "id": 2,
      "name": "sorghum",
      "supercategory": "plant"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 300, 400],  # [x, y, width, height]
      "area": 120000,
      "iscrowd": 0,
      "segmentation": []
    }
  ]
}
```

## YOLO Format

1. Annotation file example (`image1_1.txt`):
```txt
0 0.479167 0.370370 0.156250 0.370370
```
Format: `<class_id> <x_center> <y_center> <width> <height>`
- All values are normalized to [0, 1]
- `class_id` starts from 0
- `x_center, y_center`: Center coordinates of the box
- `width, height`: Box dimensions

2. Classes file example (`classes.txt`):
```txt
maize
sorghum
```
