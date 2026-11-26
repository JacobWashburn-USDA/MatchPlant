# 8 · Image-to-Ortho Projector

**Stage 4 · Locate and analyze** · **Type: Form** · [← Back to MatchPlant](../README.md)

Project detected bounding boxes from the undistorted (or tiled) test images back onto the orthomosaic, producing `projected_boxes.csv`. It uses the camera positions and the DSM from OpenDroneMap to turn each box's pixel position into map coordinates.

| | |
| --- | --- |
| **Input** | The ODM project folder from [2 ODM Runner](../2_odm_runner/README.md), and `coco_predictions.json` from [7 Object Detection Tester](../7_obj_det_tester/README.md) |
| **Output** | `projected_boxes.csv` in `<ODM project>/orthorectified2/` |
| **Previous step** | [7 Object Detection Tester](../7_obj_det_tester/README.md) |
| **Next step** | [9 Spatial Stats Extractor](../9_spatial_stats_extractor/README.md): use `projected_boxes.csv` as the CSV file |

## Run it from the dashboard

1. On the dashboard home, select **Image-to-Ortho Projector**.
2. If the page reports missing packages, click **Install requirements** and wait for it to finish.
3. Click **Browse…** to choose the required inputs:
   - **Dataset path**: your ODM project folder (the one you used in ODM Runner).
   - **Predictions json**: `coco_predictions.json` from Object Detection Tester.
4. Choose **Annotation json (optional)**: the test annotations (`annotations/test.json`) used in Object Detection Tester. This is needed to match prediction image IDs to file names, unless you provide an image list instead.
5. Keep the other defaults, or adjust them (see [Settings](#settings)).
6. Click **Run** and follow the **Output** panel.
7. Use `<ODM project>/orthorectified2/projected_boxes.csv` in [9 Spatial Stats Extractor](../9_spatial_stats_extractor/README.md).

## Inputs

**From 2 ODM Runner** (inside the ODM project folder):

```text
your_drone_project/
├── opensfm/undistorted/images/            # undistorted images
├── opensfm/undistorted/reconstruction.json  # camera parameters
├── odm_dem/dsm.tif                        # digital surface model
└── odm_georeferencing/coords.txt          # coordinate offsets
```

**From 7 Object Detection Tester:** `coco_predictions.json`. Both a flat list of predictions and a full COCO dictionary (with `images` and `annotations`) are accepted. For a flat list, also select the test annotation file.

**Tiled test images:** offsets are worked out from tile names that end in a row and column, such as `0688_548.JPG_r2c3.tif` (as made by [5 Image Splitter](../5_img_splitter/README.md)). If your tiles are named differently, provide a tile metadata JSON:

```json
{"tile.jpg": {"original_image": "DJI_0123.JPG", "offset_x": 0, "offset_y": 0}}
```

## Settings

| Field | Default | Notes |
| --- | --- | --- |
| Dataset path | (required) | ODM project folder |
| Predictions json | (required) | From Object Detection Tester |
| Annotation json (optional) | | Test COCO annotations, to map image IDs to file names |
| Image list (optional) | | Text file with one image name per line, in `image_id` order (alternative to the annotation file) |
| Tile metadata (optional) | | Only if tile names do not include `_r<row>c<col>` |
| Output dir | `orthorectified2` | Created inside the dataset path unless you give a full path |
| Dem filename | `odm_dem/dsm.tif` | DSM, relative to the dataset path |
| Worker threads | 1 | Increase only if it speeds things up; use 1 if errors occur |
| Interpolation | `bilinear` | Or `nearest` |
| Visibility test | off | More accurate (removes hidden-area artifacts) but slower |

## Outputs

`projected_boxes.csv`: one row per detection, 13 columns, **no header row**.

| Columns | Content | Example |
| --- | --- | --- |
| 1–6 | Affine transform (a, b, c, d, e, f) | `0.003, 0.0, 562340.234, ...` |
| 7–9 | Placeholders (zeros) | `0.0, 0.0, 0.0` |
| 10 | Box size in pixels `"(0, 0), (w, h)"` | `"(0, 0), (85, 120)"` |
| 11 | Box name | `DJI_0123_box_1.tif` |
| 12 | Detection ID | `1` |
| 13 | Confidence score | `0.92` |

```csv
0.003,0.0,562340.234,0.0,-0.003,4512890.567,0.0,0.0,0.0,"(0, 0), (85, 120)",DJI_0123_box_1.tif,1,0.92
```

See [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md) for details.

## Run without the dashboard

```bash
pip install -r requirements.txt
python project_boxes.py \
  --dataset-path /path/to/odm_project \
  --predictions-json /path/to/test_results/coco_predictions.json \
  --annotation-json /path/to/annotations/test.json \
  --output-dir orthorectified2 \
  --num-threads 4
```

Other options: `--image-list`, `--tile-metadata`, `--dem-filename odm_dem/dsm.tif`, `--interpolation bilinear|nearest`, `--visibility-test`. Run `python project_boxes.py --help` for the full list.

## Troubleshooting

- **"Image not found in reconstruction":** tile names must map to undistorted image names in `opensfm/undistorted/images/`; check tile names or provide tile metadata.
- **"No valid pixels found":** the box is outside the DSM; check DSM coverage and tile offsets.
- **"Cannot find coords.txt":** re-run [2 ODM Runner](../2_odm_runner/README.md) and check `odm_georeferencing/coords.txt`.
- **Errors with several threads, or out of memory:** set **Worker threads** to 1.
- **Boxes do not line up in Module 9:** check the tile names or metadata, and that the DSM and raster use the same coordinate system.

<details>
<summary>For developers</summary>

```bash
python tests/smoke_projection.py
```

Checks identical output with 1, 2, and 8 threads on a small synthetic fixture, and a nonzero exit for missing inputs.

</details>

## License

MIT License. See [LICENSE](../LICENSE).
