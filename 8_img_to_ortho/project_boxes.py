"""
Script Name: project_boxes.py
Purpose: To project detected bounding boxes from undistorted UAV images onto orthomosaic using forward projection
Date Created: January 27, 2026
Version: 1.2

Credits:
    Original forward projection algorithm by Piero Toffanin (OpenDroneMap)
    Adapted for MatchPlant pipeline by Worasit Sangjan, Piyush Pandey
"""
import os
import sys
sys.path.insert(0, os.path.join("..", "..", os.path.dirname(__file__)))

from math import sqrt
import rasterio
import numpy as np
import numpy.ma as ma
import multiprocessing
from skimage.draw import line
import cv2
import json
import pandas as pd
import logging
import argparse
import re

# =====================================================================
# Configuration - Edit these paths for your project
# =====================================================================
# IMPORTANT: Change these paths before running!

# Path to your ODM dataset folder
dataset_path = '/path/to/your/odm_dataset'  # ← CHANGE THIS!

# Path to predictions.json from Module 7 (Object Detection Testing)
predictions_json_path = "test_results/coco_predictions.json"  # ← CHANGE THIS!

# Optional: Path to tile metadata from Module 5 (if you used tiled images)
# Format: {"tile_001.jpg": {"original_image": "DJI_0123.JPG", "offset_x": 0, "offset_y": 0}, ...}
tile_metadata_path = None  # Set to path if you have it, else keep as None

# Optional: COCO annotation file from Module 7 testing. Use this when predictions are saved as a flat COCO annotations list.
annotation_json_path = None

# Optional: one image filename per line, matching image_id order.
image_list_path = None

# Output settings
output_dir = "orthorectified2"  # Folder name for output (created inside dataset_path)
dem_filename = "odm_dem/dsm.tif"  # Path to DSM relative to dataset_path

# Processing settings
num_threads = multiprocessing.cpu_count()  # Use all CPU cores (change if needed)
interpolation_method = 'bilinear'  # Options: 'bilinear' or 'nearest'
skip_visibility_test = True  # Set to False for more accurate (but slower) results

# =====================================================================

def parse_args():
    """Parse optional command-line overrides while preserving edit-in-file usage."""
    parser = argparse.ArgumentParser(
        description="Project Module 7 object detections onto an ODM orthomosaic."
    )
    parser.add_argument("--dataset-path", default=dataset_path,
                        help="ODM project folder containing opensfm/, odm_dem/, and odm_georeferencing/.")
    parser.add_argument("--predictions-json", default=predictions_json_path,
                        help="Module 7 prediction JSON. Supports full COCO dict or flat COCO prediction list.")
    parser.add_argument("--annotation-json", default=annotation_json_path,
                        help="Optional Module 7 COCO test annotation JSON for image_id to filename mapping.")
    parser.add_argument("--image-list", default=image_list_path,
                        help="Optional text file with one image filename per line, matching image_id order.")
    parser.add_argument("--tile-metadata", default=tile_metadata_path,
                        help="Optional tile metadata JSON from Module 5.")
    parser.add_argument("--output-dir", default=output_dir,
                        help="Output folder name, created inside dataset path unless absolute.")
    parser.add_argument("--dem-filename", default=dem_filename,
                        help="DSM path relative to dataset path.")
    parser.add_argument("--num-threads", type=int, default=num_threads,
                        help="CPU worker count. Use 1 if multiprocessing causes platform issues.")
    parser.add_argument("--interpolation", choices=["bilinear", "nearest"], default=interpolation_method,
                        help="Pixel interpolation method.")
    parser.add_argument("--visibility-test", action="store_true",
                        help="Enable visibility testing. More accurate but slower.")
    return parser.parse_args()


ARGS = parse_args()

dataset_path = ARGS.dataset_path
predictions_json_path = ARGS.predictions_json
annotation_json_path = ARGS.annotation_json
image_list_path = ARGS.image_list
tile_metadata_path = ARGS.tile_metadata
output_dir = ARGS.output_dir
dem_filename = ARGS.dem_filename
num_threads = max(1, ARGS.num_threads)
interpolation_method = ARGS.interpolation
skip_visibility_test = not ARGS.visibility_test

# Setup logging
logging.basicConfig(filename='output.log', encoding='utf-8', level=logging.INFO)

# Construct full paths
dem_path = os.path.join(dataset_path, dem_filename)
output_path = output_dir if os.path.isabs(output_dir) else os.path.join(dataset_path, output_dir)

print("="*60)
print("MatchPlant Module 8: Bounding Box Projection")
print("="*60)
print(f"Dataset path: {dataset_path}")
print(f"Predictions: {predictions_json_path}")
print(f"DEM path: {dem_path}")
print(f"Output directory: {output_path}")
print(f"Using {num_threads} CPU threads")
print("="*60)

# =====================================================================
# Helper Functions
# =====================================================================

def load_image_list(list_path):
    """Load image filenames from a text file using zero-based line number as image_id."""
    images = []
    if list_path and os.path.exists(list_path):
        with open(list_path, 'r') as f:
            for idx, line in enumerate(f):
                filename = line.strip()
                if filename:
                    images.append({"id": idx, "file_name": filename})
        print(f"Loaded {len(images)} image names from {list_path}")
    return images


def load_annotation_images(annotation_path):
    """Load image metadata from the COCO annotation file used for Module 7 testing."""
    if not annotation_path or not os.path.exists(annotation_path):
        return []

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    images = annotations.get("images", [])
    if images:
        print(f"Loaded {len(images)} image names from {annotation_path}")
    return images


def load_predictions(predictions_path, annotation_path=None, list_path=None):
    """
    Load COCO format predictions from Module 7 output.

    Supports either a full COCO-style dict with images/categories/annotations
    or a flat list of COCO prediction annotations from pycocotools/test.py.
    """
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    if isinstance(predictions, list):
        predictions = {
            "images": load_annotation_images(annotation_path) or load_image_list(list_path),
            "categories": [],
            "annotations": predictions,
        }
    else:
        if "annotations" not in predictions:
            raise ValueError("Predictions JSON must contain 'annotations' or be a flat prediction list")
        if not predictions.get("images"):
            predictions["images"] = load_annotation_images(annotation_path) or load_image_list(list_path)

    if not predictions.get("images"):
        raise ValueError(
            "No image metadata found. Provide a full COCO predictions file, --annotation-json, or --image-list."
        )

    print(f"Loaded {len(predictions['annotations'])} predictions from {len(predictions['images'])} test images")
    return predictions


def load_tile_metadata(metadata_path=None):
    """
    Load tile metadata if available (created by Module 5)
    Returns: dict mapping tile_filename -> {original_image, offset_x, offset_y}
    """
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            print(f"✓ Loaded tile metadata from {metadata_path}")
            return json.load(f)
    else:
        print("No external tile metadata found. Deriving tile offsets from filenames when possible.")
        return {}


def resolve_undistorted_filename(shot_fname, odm_dataset_path):
    """Try common filename variants to match ODM undistorted image files."""
    candidates = [shot_fname]
    base, ext = os.path.splitext(shot_fname)

    if ext.lower() == '.jpg':
        candidates.append(shot_fname + '.tif')
    elif ext.lower() == '.tif' and not shot_fname.lower().endswith('.jpg.tif'):
        candidates.append(base + '.JPG.tif')

    if not shot_fname.upper().startswith('DJI_'):
        candidates.append('DJI_' + shot_fname)
        if ext.lower() == '.jpg':
            candidates.append('DJI_' + shot_fname + '.tif')

    if shot_fname.upper().startswith('DJI_') and ext.lower() == '.jpg':
        candidates.append(shot_fname + '.tif')

    for candidate in candidates:
        candidate_path = os.path.join(odm_dataset_path, 'opensfm', 'undistorted', 'images', candidate)
        if os.path.exists(candidate_path):
            return candidate
    return shot_fname


def original_image_from_tile_name(tile_name):
    """Convert common tile names like 0688_548.JPG_r2c3.tif to DJI_0688_548.JPG.tif."""
    original_image = tile_name.replace('test_', '')
    original_image = re.sub(r'_r\d+c\d+\.tif$', '.tif', original_image)
    original_image = re.sub(r'_tile_\d+.*$', '.JPG', original_image)
    if not original_image.upper().startswith('DJI_'):
        original_image = 'DJI_' + original_image
    return original_image


def tile_offsets_from_name(tile_name, image_info=None):
    """Derive tile offsets from row/column filenames when no metadata JSON is present."""
    match = re.search(r'_r(\d+)c(\d+)\.tif$', tile_name)
    if not match:
        return 0, 0

    row = int(match.group(1))
    col = int(match.group(2))
    tile_info = (image_info or {}).get('tile_info', {})

    grid_size = tile_info.get('grid_size', '6x4')
    try:
        grid_cols, grid_rows = map(int, grid_size.lower().split('x'))
    except ValueError:
        grid_cols, grid_rows = 6, 4

    original_w = int(tile_info.get('original_width', 5472))
    original_h = int(tile_info.get('original_height', 3648))
    context = int(tile_info.get('overlap_pixels', tile_info.get('context_pixels', 100)))

    cell_w = original_w / grid_cols
    cell_h = original_h / grid_rows

    offset_x = (col - 1) * cell_w
    offset_y = (row - 1) * cell_h
    if col > 1:
        offset_x -= context
    if row > 1:
        offset_y -= context

    return offset_x, offset_y


def tile_to_original_coords(bbox, tile_metadata=None, tile_name=None):
    """
    Convert bounding box from tile coordinates to original image coordinates
    
    Args:
        bbox: [x, y, width, height] in COCO format (tile coordinates)
        tile_metadata: dict from load_tile_metadata()
        tile_name: filename of the tile image
    
    Returns:
        [x, y, width, height] in original image coordinates
        original_image_name: name of the original undistorted image
    """
    x, y, w, h = bbox
    
    # If we have tile metadata, use it
    if tile_metadata and tile_name in tile_metadata:
        meta = tile_metadata[tile_name]
        offset_x = meta.get('offset_x', 0)
        offset_y = meta.get('offset_y', 0)
        original_image = meta.get('original_image', tile_name)
    else:
        offset_x, offset_y = tile_offsets_from_name(tile_name)
        original_image = original_image_from_tile_name(tile_name)
    
    # Transform coordinates
    bbox_original = [x + offset_x, y + offset_y, w, h]
    
    return bbox_original, original_image


def read_reconstruction_json(dataset_path):
    """Read the reconstruction.json file"""
    recon_file = open(dataset_path + "/opensfm/undistorted/reconstruction.json")
    reconstructions = json.load(recon_file)
    recon_file.close()
    return reconstructions


def extract_shots_from_reconstruction(reconstruction_dict):
    reconstruction = reconstruction_dict[0]
    shots = reconstruction['shots']
    return shots


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# =====================================================================
# Main Processing Function
# =====================================================================

def main():
    """Main processing function"""
    
    # Read DEM
    print("\nReading DEM: %s" % dem_path)
    with rasterio.open(dem_path) as dem_raster:
        dem = dem_raster.read()[0]
        dem_has_nodata = dem_raster.profile.get('nodata') is not None

        if dem_has_nodata:
            m = ma.array(dem, mask=dem==dem_raster.nodata)
            dem_min_value = m.min()
            dem_max_value = m.max()
        else:
            dem_min_value = dem.min()
            dem_max_value = dem.max()

        print("DEM Minimum: %s" % dem_min_value)
        print("DEM Maximum: %s" % dem_max_value)
        
        h, w = dem.shape

        crs = dem_raster.profile.get('crs')
        dem_offset_x, dem_offset_y = (0, 0)

        if crs:
            print("DEM has a CRS: %s" % str(crs))
            coords_file = os.path.join(dataset_path, "odm_georeferencing", "coords.txt")
            if not os.path.exists(coords_file):
                print("ERROR: Cannot find %s (we need that!)" % coords_file)
                return
            
            with open(coords_file) as f:
                l = f.readline()
                l = f.readline().rstrip()
                dem_offset_x, dem_offset_y = map(float, l.split(" "))
            
            print("DEM offset: (%s, %s)" % (dem_offset_x, dem_offset_y))

        print("DEM dimensions: %sx%s pixels" % (w, h))

        # Read reconstruction
        print("\nReading camera reconstruction...")
        reconstructions = read_reconstruction_json(dataset_path)
        
        if len(reconstructions) == 0:
            raise Exception("ERROR: No reconstructions available")

        print(f"Using {num_threads} CPU threads")

        shots = extract_shots_from_reconstruction(reconstructions)
        print(f"Found {len(shots)} camera shots")
        
        # Load predictions
        print("\nLoading predictions...")
        predictions = load_predictions(predictions_json_path, annotation_json_path, image_list_path)
        tile_metadata = load_tile_metadata(tile_metadata_path)
        
        # Create image_id to filename mapping
        image_id_to_name = {img['id']: img['file_name'] for img in predictions['images']}
        
        # Group annotations by image for processing
        annotations_by_image = {}
        for ann in predictions['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        print(f"Processing {len(annotations_by_image)} images with detections")
        
        # Create output directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory: {output_path}")
        
        transform_list = []
        csv_name = 'projected_boxes.csv'
        
        # Process each image with detections
        print("\n" + "="*60)
        print("Starting projection...")
        print("="*60)
        
        for image_id, annotations in annotations_by_image.items():
            tile_filename = image_id_to_name[image_id]
            print(f"\n{'='*60}")
            print(f"Processing: {tile_filename} ({len(annotations)} detections)")
            print(f"{'='*60}")
            
            # Process each detection in this image
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, w, h] in tile coordinates
                score = ann.get('score', 1.0)
                detection_id = ann.get('id', f"{ann.get('image_id', 'img')}_{len(transform_list)}")
                
                # Convert to original image coordinates
                bbox_original, original_image_name = tile_to_original_coords(
                    bbox, tile_metadata, tile_filename
                )
                
                print(f"  Detection {detection_id}: score={score:.3f}, bbox={bbox} -> {bbox_original}")
                
                # Check if shot exists in reconstruction
                shot_fname = resolve_undistorted_filename(original_image_name, dataset_path)
                box_fname = f"{os.path.splitext(shot_fname)[0]}_box_{detection_id}.tif"
                
                if shot_fname not in shots:
                    print(f"    WARNING: {shot_fname} not found in reconstruction, skipping")
                    continue
                
                # Get the path to the undistorted image
                undistorted_image_path = os.path.join(dataset_path, 'opensfm', 'undistorted', 'images', shot_fname)
                
                if not os.path.exists(undistorted_image_path):
                    print(f"    WARNING: {undistorted_image_path} does not exist, skipping")
                    continue
                
                # Load the shot image
                shot_image = cv2.imread(undistorted_image_path, cv2.IMREAD_UNCHANGED)
                num_bands = 3 if len(shot_image.shape) == 3 else 1
                
                # Get image dimensions
                img_h, img_w = shot_image.shape[:2]
                half_img_w = img_w / 2.0
                half_img_h = img_h / 2.0
                
                # Get camera parameters from shot
                shot = shots[shot_fname]
                cam_id = shot['camera']
                
                # Get focal length
                cam = reconstructions[0]['cameras'][cam_id]
                f = cam['focal'] * max(img_w, img_h)
                
                # Get camera position
                Xs, Ys, Zs = shot['translation']
                
                # Get rotation matrix
                rotation = shot['rotation']
                if len(rotation) == 3 and not isinstance(rotation[0], (list, tuple)):
                    rotation, _ = cv2.Rodrigues(np.array(rotation, dtype=np.float64))
                a1, a2, a3 = rotation[0]
                b1, b2, b3 = rotation[1]
                c1, c2, c3 = rotation[2]
                
                # Calculate camera grid position for visibility testing
                cam_x, cam_y = dem_raster.index(Xs + dem_offset_x, Ys + dem_offset_y)
                cam_grid_x = max(0, min(w - 1, cam_x))
                cam_grid_y = max(0, min(h - 1, cam_y))
                
                # Distance map for visibility
                distance_map = np.hypot(np.arange(w)[np.newaxis,:] - cam_grid_x,
                                       np.arange(h)[:,np.newaxis] - cam_grid_y)
                
                # Calculate DEM bounding box for this detection
                x, y, bw, bh = bbox_original
                
                # Define the 4 corners of the bounding box in image coordinates
                bbox_corners_img = [
                    [x - half_img_w, y - half_img_h],           # top-left
                    [x + bw - half_img_w, y - half_img_h],      # top-right
                    [x + bw - half_img_w, y + bh - half_img_h], # bottom-right
                    [x - half_img_w, y + bh - half_img_h]       # bottom-left
                ]
                
                # Project bbox corners to DEM coordinates
                def dem_coordinates_from_img(cpx, cpy):
                    """Project image point to DEM using minimum elevation"""
                    Za = dem_min_value
                    m = (a3*b1*cpy - a1*b3*cpy - (a3*b2 - a2*b3)*cpx - (a2*b1 - a1*b2)*f)
                    Xa = dem_offset_x + (m*Xs + (b3*c1*cpy - b1*c3*cpy - (b3*c2 - b2*c3)*cpx - (b2*c1 - b1*c2)*f)*Za - (b3*c1*cpy - b1*c3*cpy - (b3*c2 - b2*c3)*cpx - (b2*c1 - b1*c2)*f)*Zs)/m
                    Ya = dem_offset_y + (m*Ys - (a3*c1*cpy - a1*c3*cpy - (a3*c2 - a2*c3)*cpx - (a2*c1 - a1*c2)*f)*Za + (a3*c1*cpy - a1*c3*cpy - (a3*c2 - a2*c3)*cpx - (a2*c1 - a1*c2)*f)*Zs)/m
                    y, x = dem_raster.index(Xa, Ya)
                    return (x, y)
                
                dem_bbox_coords = [dem_coordinates_from_img(corner[0], corner[1]) for corner in bbox_corners_img]
                dem_bbox_x = np.array([coord[0] for coord in dem_bbox_coords])
                dem_bbox_y = np.array([coord[1] for coord in dem_bbox_coords])
                
                dem_bbox_minx = min(w - 1, max(0, int(dem_bbox_x.min())))
                dem_bbox_miny = min(h - 1, max(0, int(dem_bbox_y.min())))
                dem_bbox_maxx = min(w - 1, max(0, int(dem_bbox_x.max())))
                dem_bbox_maxy = min(h - 1, max(0, int(dem_bbox_y.max())))
                
                dem_bbox_w = 1 + dem_bbox_maxx - dem_bbox_minx
                dem_bbox_h = 1 + dem_bbox_maxy - dem_bbox_miny
                
                print(f"    DEM box: [({dem_bbox_minx}, {dem_bbox_miny}), ({dem_bbox_maxx}, {dem_bbox_maxy})] ({dem_bbox_w}x{dem_bbox_h} pixels)")
                
                has_nodata = dem_raster.profile.get('nodata') is not None
                
                # Forward projection
                def process_pixels(step):
                    imgout = np.full((num_bands, dem_bbox_h, dem_bbox_w), np.nan)
                    
                    minx = dem_bbox_w
                    miny = dem_bbox_h
                    maxx = 0
                    maxy = 0
                    
                    for j in range(dem_bbox_miny, dem_bbox_maxy + 1):
                        if j % num_threads == step:
                            im_j = j - dem_bbox_miny
                            
                            for i in range(dem_bbox_minx, dem_bbox_maxx + 1):
                                im_i = i - dem_bbox_minx
                                
                                # World coordinates
                                Za = dem[j][i]
                                
                                # Skip nodata
                                if has_nodata and Za == dem_raster.nodata:
                                    continue
                                
                                Xa, Ya = dem_raster.xy(j, i)
                                
                                # Remove offset
                                Xa -= dem_offset_x
                                Ya -= dem_offset_y
                                
                                # Colinearity function
                                dx = (Xa - Xs)
                                dy = (Ya - Ys)
                                dz = (Za - Zs)
                                
                                den = a3 * dx + b3 * dy + c3 * dz
                                x_proj = half_img_w - (f * (a1 * dx + b1 * dy + c1 * dz) / den)
                                y_proj = half_img_h - (f * (a2 * dx + b2 * dy + c2 * dz) / den)
                                
                                if x_proj >= 0 and y_proj >= 0 and x_proj <= img_w - 1 and y_proj <= img_h - 1:
                                    # Visibility test (optional)
                                    if not skip_visibility_test:
                                        check_dem_points = np.column_stack(line(i, j, cam_grid_x, cam_grid_y))
                                        check_dem_points = check_dem_points[np.all(np.logical_and(np.array([0, 0]) <= check_dem_points, check_dem_points < [w, h]), axis=1)]
                                        
                                        visible = True
                                        for p in check_dem_points:
                                            ray_z = Zs + (distance_map[p[1]][p[0]] / distance_map[j][i]) * dz
                                            if ray_z > dem_max_value:
                                                break
                                            if dem[p[1]][p[0]] > ray_z:
                                                visible = False
                                                break
                                        if not visible:
                                            continue
                                    
                                    # Interpolate pixel value
                                    if interpolation_method == 'bilinear':
                                        xi = img_w - 1 - x_proj
                                        yi = img_h - 1 - y_proj
                                        values = bilinear_interpolate(shot_image, xi, yi)
                                    else:
                                        xi = img_w - 1 - int(round(x_proj))
                                        yi = img_h - 1 - int(round(y_proj))
                                        values = shot_image[yi][xi]
                                    
                                    # Skip all-zero values
                                    if not np.all(values == 0):
                                        minx = min(minx, im_i)
                                        miny = min(miny, im_j)
                                        maxx = max(maxx, im_i)
                                        maxy = max(maxy, im_j)
                                        
                                        for b in range(num_bands):
                                            imgout[b][im_j][im_i] = values[b]
                    
                    return (imgout, (minx, miny, maxx, maxy))
                
                # Run processing (multiprocessing)
                if num_threads > 1:
                    with multiprocessing.Pool(num_threads) as p:
                        try:
                            results = p.map(process_pixels, range(num_threads))
                        except:
                            logging.info(f"Multithreading error for {box_fname}")
                            continue
                else:
                    results = [process_pixels(0)]
                
                results = list(filter(lambda r: r[1][0] <= r[1][2] and r[1][1] <= r[1][3], results))
                
                if len(results) == 0:
                    print(f"    WARNING: No valid pixels found for detection {detection_id}")
                    continue
                
                # Merge results
                try:
                    imgout, _ = results[0]
                except:
                    logging.info(f"imgout error for {box_fname}")
                    continue
                
                for j in range(dem_bbox_miny, dem_bbox_maxy + 1):
                    try:
                        im_j = j - dem_bbox_miny
                        resimg, _ = results[j % num_threads]
                        for b in range(num_bands):
                            imgout[b][im_j] = resimg[b][im_j]
                    except:
                        logging.info(f"res_img merge error for {box_fname}")
                        continue
                
                # Merge bounds
                minx = dem_bbox_w
                miny = dem_bbox_h
                maxx = 0
                maxy = 0
                
                for _, bounds in results:
                    minx = min(bounds[0], minx)
                    miny = min(bounds[1], miny)
                    maxx = max(bounds[2], maxx)
                    maxy = max(bounds[3], maxy)
                
                print(f"    Output bounds: ({minx}, {miny}), ({maxx}, {maxy}) pixels")
                
                if minx <= maxx and miny <= maxy:
                    imgout = imgout[:,miny:maxy+1,minx:maxx+1]
                    
                    dem_transform = dem_raster.profile['transform']
                    offset_x, offset_y = dem_raster.xy(dem_bbox_miny + miny, dem_bbox_minx + minx, offset='ul')
                    
                    profile = {
                        'driver': 'GTiff',
                        'count': num_bands,
                        'transform': rasterio.transform.Affine(dem_transform[0], dem_transform[1], offset_x,
                                                               dem_transform[3], dem_transform[4], offset_y),
                        'dim': list(imgout.shape),
                        'nodata': None,
                        'crs': crs
                    }
                    
                    # Save transform info in Module 9 compatible format
                    # Columns 0-5: Affine transform (a, b, c, d, e, f)
                    affine = profile['transform']
                    transform1 = [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f]
                    
                    # Columns 6-8: Three zeros (required by Module 9)
                    transform1.extend([0.0, 0.0, 0.0])
                    
                    # Column 9: Dimensions as string "(0, 0), (width, height)"
                    dims = profile['dim']  # [bands, height, width]
                    dim_string = f"(0, 0), ({dims[2]}, {dims[1]})"
                    transform1.append(dim_string)
                    
                    # Column 10: Filename
                    transform1.append(box_fname)
                    
                    # Columns 11-12: Optional metadata for reference
                    transform1.append(detection_id)
                    transform1.append(score)
                    
                    transform_list.append(transform1)
                    
                    # Save CSV incrementally WITHOUT HEADERS (Module 9 expects no headers)
                    df1 = pd.DataFrame(transform_list)
                    outfile = os.path.join(output_path, csv_name)
                    df1.to_csv(outfile, index=False, header=False)
                    
                    print(f"    Saved projection info to {csv_name}")

    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results saved to: {output_path}/{csv_name}")
    print("="*60)

# =====================================================================
# Run this Script
# =====================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR occurred during processing:")
        print(str(e))
        print("="*60)
        print("\nPlease check:")
        print("1. All paths in the configuration section are correct")
        print("2. ODM processing has been completed (Module 2)")
        print("3. predictions.json exists from Module 7")
        print("4. All required files are in the correct locations")
        import traceback
        traceback.print_exc()