"""
Module manifest for the MatchPlant web dashboard.

Each entry describes one pipeline module: where its script(s) live, whether it
launches a native (Tkinter) GUI or is a CLI/config-driven tool, and, for CLI
tools, the fields to render as a form so the script can be run with real
arguments instead of hand-typing a command.
"""

import settings

GITHUB_REPO = "https://github.com/JacobWashburn-USDA/MatchPlant"
GITHUB_BRANCH = "main"

# Field kinds understood by the form renderer: "text", "int", "float",
# "select", "checkbox".


def _train_fields():
    return [
        {"name": "seed", "flag": "--seed", "kind": "int", "help": "Seed for one independent training run."},
        {"name": "run_name", "flag": "--run_name", "kind": "text", "help": "Run name. Used as runs/<run_name> when Output dir is not set."},
        {"name": "output_dir", "flag": "--output_dir", "kind": "text", "help": "Run output directory for checkpoints and validation logs."},
        {"name": "train_data_dir", "flag": "--train-data-dir", "kind": "text", "default": "data/train", "help": "Fixed training tile directory."},
        {"name": "val_data_dir", "flag": "--val-data-dir", "kind": "text", "default": "data/val", "help": "Fixed validation tile directory."},
        {"name": "train_annotation_file", "flag": "--train-annotation-file", "kind": "text", "default": "annotations/train.json", "help": "Fixed training COCO JSON."},
        {"name": "val_annotation_file", "flag": "--val-annotation-file", "kind": "text", "default": "annotations/val.json", "help": "Fixed validation COCO JSON."},
        {"name": "epochs", "flag": "--epochs", "kind": "int", "help": "Number of training epochs."},
        {"name": "batch_size", "flag": "--batch-size", "kind": "int", "help": "Images per training batch."},
        {"name": "num_workers", "flag": "--num-workers", "kind": "int", "help": "DataLoader workers. Defaults to an automatic value."},
        {"name": "val_frequency", "flag": "--val-frequency", "kind": "int", "default": "1", "help": "Run validation every N epochs."},
        {"name": "early_stopping_patience", "flag": "--early-stopping-patience", "kind": "int", "help": "Stop after this many validations without improvement. Disabled by default."},
        {"name": "lr", "flag": "--lr", "kind": "float", "default": "0.05", "help": "SGD learning rate."},
        {"name": "no_amp", "flag": "--no-amp", "kind": "checkbox", "help": "Disable CUDA mixed precision training."},
        {"name": "no_pin_memory", "flag": "--no-pin-memory", "kind": "checkbox", "help": "Disable pinned-memory DataLoader transfer on CUDA."},
        {"name": "clear_cache_each_step", "flag": "--clear-cache-each-step", "kind": "checkbox", "help": "Clear CUDA cache each step. Slower, but can help debug memory pressure."},
        {"name": "best_metric", "flag": "--best-metric", "kind": "select", "choices": ["ap50", "map50_95"], "default": "ap50", "help": "Validation metric used for best checkpoint selection."},
    ]


def _test_fields():
    return [
        {"name": "data_dir", "flag": "--data-dir", "kind": "text", "default": "data/test", "help": "Directory containing test tiles."},
        {"name": "annotation_file", "flag": "--annotation-file", "kind": "text", "default": "annotations/test.json", "help": "COCO annotation JSON for the test split."},
        {"name": "checkpoint", "flag": "--checkpoint", "kind": "text", "default": "checkpoints/best_model.pt", "help": "Model checkpoint to evaluate."},
        {"name": "results_dir", "flag": "--results-dir", "kind": "text", "default": "paper_results", "help": "Directory for output metrics and plots."},
    ]


def _project_boxes_fields():
    return [
        {"name": "dataset_path", "flag": "--dataset-path", "kind": "text", "help": "ODM project folder containing opensfm/, odm_dem/, and odm_georeferencing/."},
        {"name": "predictions_json", "flag": "--predictions-json", "kind": "text", "help": "Module 7 prediction JSON (full COCO dict or flat COCO prediction list)."},
        {"name": "annotation_json", "flag": "--annotation-json", "kind": "text", "help": "Optional Module 7 COCO test annotation JSON for image_id to filename mapping."},
        {"name": "image_list", "flag": "--image-list", "kind": "text", "help": "Optional text file with one image filename per line, matching image_id order."},
        {"name": "tile_metadata", "flag": "--tile-metadata", "kind": "text", "help": "Optional tile metadata JSON from Module 5."},
        {"name": "output_dir", "flag": "--output-dir", "kind": "text", "help": "Output folder name, created inside dataset path unless absolute."},
        {"name": "dem_filename", "flag": "--dem-filename", "kind": "text", "help": "DSM path relative to dataset path."},
        {"name": "num_threads", "flag": "--num-threads", "kind": "int", "help": "CPU worker count. Use 1 if multiprocessing causes platform issues."},
        {"name": "interpolation", "flag": "--interpolation", "kind": "select", "choices": ["bilinear", "nearest"], "default": "bilinear", "help": "Pixel interpolation method."},
        {"name": "visibility_test", "flag": "--visibility-test", "kind": "checkbox", "help": "Enable visibility testing. More accurate but slower."},
    ]


MODULES = [
    {
        "id": "0_gps_embeder",
        "stage": "1. Data Preprocessing",
        "title": "GPS Embedder",
        "dir": "0_gps_embeder",
        "description": "Embed GPS coordinates into UAV image EXIF metadata before orthomosaic generation.",
        "kind": "gui",
        "script": {"any": "gps_embed.py"},
        "fields": [],
    },
    {
        "id": "1_gcp_finder",
        "stage": "1. Data Preprocessing",
        "title": "GCP Finder",
        "dir": "1_gcp_finder",
        "description": "Interactive tool to locate Ground Control Points (GCPs) in UAV imagery and build a gcp_list.txt.",
        "kind": "gui",
        "script": {"mac": "gcp_finder_mac.py", "win": "gcp_finder_win.py"},
        "fields": [],
    },
    {
        "id": "2_odm_runner",
        "stage": "1. Data Preprocessing",
        "title": "ODM Runner",
        "dir": "2_odm_runner",
        "description": "Runs OpenDroneMap (via Docker) on a project folder to generate the orthomosaic and DSM. Requires Docker Desktop.",
        "kind": "shell",
        "script": {"mac": "run_ODM_process.sh", "win": "run_ODM_process.ps1"},
        "fields": [
            {"name": "project_folder", "kind": "cwd", "label": "ODM project folder", "help": "Folder containing an images/ subfolder (and optional gcp_list.txt). The script is run with this as its working directory."},
        ],
    },
    {
        "id": "3_min_img_finder",
        "stage": "2. Data Preparation",
        "title": "Minimum Image Finder",
        "dir": "3_min_img_finder",
        "description": "Select the smallest set of UAV images that still covers all plants of interest, to minimize labeling effort.",
        "kind": "gui",
        "script": {"mac": "min_img_finder_mac.py", "win": "min_img_finder_win.py"},
        "fields": [],
    },
    {
        "id": "4_bbox_drawer",
        "stage": "2. Data Preparation",
        "title": "BBox Drawer",
        "dir": "4_bbox_drawer",
        "description": "Interactive bounding-box annotation GUI producing COCO-format labels.",
        "kind": "gui",
        "script": {"mac": "bbox_drawer_mac.py", "win": "bbox_drawer_win.py"},
        "fields": [],
    },
    {
        "id": "5_img_splitter",
        "stage": "2. Data Preparation",
        "title": "Image Splitter",
        "dir": "5_img_splitter",
        "description": "Tile large UAV images and their COCO annotations into train/val/test splits for model training.",
        "kind": "gui",
        "script": {"mac": "img_spliter_mac.py", "win": "img_splitter_win.py"},
        "fields": [],
    },
    {
        "id": "6-1_obj_det_trainer",
        "stage": "3. Model Development",
        "title": "Object Detection Trainer",
        "dir": "6-1_obj_det_trainer",
        "description": "Train a Faster R-CNN model on tiled COCO datasets produced by the Image Splitter.",
        "kind": "cli",
        "script": {"any": "train.py"},
        "fields": _train_fields(),
    },
    {
        "id": "6-2_obj_det_trans_learner",
        "stage": "3. Model Development",
        "title": "Transfer Learning",
        "dir": "6-2_obj_det_trans_learner",
        "description": "Fine-tune a pretrained Faster R-CNN checkpoint on a new dataset. Configured via transfer_config.yaml.",
        "kind": "config",
        "script": {"any": "transfer_train.py"},
        "config_file": "transfer_config.yaml",
        "fields": [],
    },
    {
        "id": "7_obj_det_tester",
        "stage": "3. Model Development",
        "title": "Object Detection Tester",
        "dir": "7_obj_det_tester",
        "description": "Evaluate a trained checkpoint against a held-out test split and report detection metrics.",
        "kind": "cli",
        "script": {"any": "test.py"},
        "fields": _test_fields(),
    },
    {
        "id": "8_img_to_ortho",
        "stage": "4. Utilization",
        "title": "Image-to-Ortho Projector",
        "dir": "8_img_to_ortho",
        "description": "Project detected bounding boxes from tiled images back onto the orthomosaic, producing projected_boxes.csv.",
        "kind": "cli",
        "script": {"any": "project_boxes.py"},
        "fields": _project_boxes_fields(),
    },
    {
        "id": "9_spatial_stats_extractor",
        "stage": "4. Utilization",
        "title": "Spatial Stats Extractor",
        "dir": "9_spatial_stats_extractor",
        "description": "Interactive tool to extract per-plant spatial statistics and shapefiles from projected detections.",
        "kind": "gui",
        "script": {"mac": "stat_extractor_mac.py", "win": "stat_extractor_win.py"},
        "fields": [],
    },
]

MODULES_BY_ID = {m["id"]: m for m in MODULES}


def module_dir(module):
    return settings.get_repo_root() / module["dir"]


def requirements_path(module, variant=None):
    """Resolve this module's requirements file, preferring the OS-specific
    variant (requirements_mac.txt / requirements_win.txt) when one exists."""
    d = module_dir(module)
    names = [f"requirements_{variant}.txt"] if variant else []
    names += ["requirements.txt", "requirements_mac.txt", "requirements_win.txt"]
    for name in names:
        f = d / name
        if f.exists():
            return f
    return None


def requirements_text(module, variant=None):
    f = requirements_path(module, variant)
    if f is None:
        return None, None
    return f.name, f.read_text()


def readme_path(module):
    f = module_dir(module) / "README.md"
    return f if f.exists() else None


def github_url(module, filename=None):
    """Link to this module's folder on GitHub, or a specific file in it."""
    path = module["dir"] if filename is None else f"{module['dir']}/{filename}"
    kind = "blob" if filename else "tree"
    return f"{GITHUB_REPO}/{kind}/{GITHUB_BRANCH}/{path}"


def github_links(module, req_name=None):
    """(label, url) pairs for every source file relevant to this module,
    so a user can inspect the real code before running it."""
    links = [("Folder", github_url(module))]
    if readme_path(module) is not None:
        links.append(("README", github_url(module, "README.md")))
    for variant, filename in module["script"].items():
        label = "Script" if variant == "any" else f"Script ({variant})"
        links.append((label, github_url(module, filename)))
    if req_name:
        links.append((req_name, github_url(module, req_name)))
    if module["kind"] == "config":
        links.append((module["config_file"], github_url(module, module["config_file"])))
    return links
