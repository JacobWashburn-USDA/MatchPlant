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
    return [{'name': 'seed',
      'flag': '--seed',
      'kind': 'int',
      'help': 'Controls random choices during training. Use the same seed when comparing experiments; exact '
              'results can still depend on hardware.',
      'label': 'Random seed (optional)',
      'default_hint': 'No fixed seed. Leave blank, or enter an integer such as 42.'},
     {'name': 'run_name',
      'flag': '--run_name',
      'kind': 'text',
      'help': 'Name this training experiment, for example maize_test_01. Results are saved in '
              'runs/maize_test_01/ inside the training module. If you choose a results folder below, that folder '
              'takes priority. If both fields are blank, results go into checkpoints/ and validation_results/ '
              'inside the training module.',
      'label': 'Experiment name (optional)',
      'default_hint': 'Example: maize_test_01'},
     {'name': 'output_dir',
      'picker': 'folder',
      'flag': '--output_dir',
      'kind': 'text',
      'help': 'Choose where to save model checkpoints and validation results. This folder takes priority over '
              'the experiment name. Browse for an existing folder, or type a new folder path. If both fields are '
              'blank, results go into checkpoints/ and validation_results/ inside the training module.',
      'label': 'Save results to folder (optional)',
      'default_hint': 'Choose this instead of an experiment name to set a specific save location.'},
     {'name': 'train_data_dir',
      'picker': 'folder',
      'flag': '--train-data-dir',
      'kind': 'text',
      'default': 'data/train',
      'help': 'Folder containing the training image tiles.',
      'label': 'Training images folder',
      'default_hint': 'Default: data/train.'},
     {'name': 'val_data_dir',
      'picker': 'folder',
      'flag': '--val-data-dir',
      'kind': 'text',
      'default': 'data/val',
      'help': 'Folder containing the validation image tiles, used to evaluate training progress.',
      'label': 'Validation images folder',
      'default_hint': 'Default: data/val.'},
     {'name': 'train_annotation_file',
      'picker': 'file',
      'flag': '--train-annotation-file',
      'kind': 'text',
      'default': 'annotations/train.json',
      'help': 'COCO JSON annotations describing the training images and bounding boxes.',
      'label': 'Training annotations file',
      'default_hint': 'Default: annotations/train.json.'},
     {'name': 'val_annotation_file',
      'picker': 'file',
      'flag': '--val-annotation-file',
      'kind': 'text',
      'default': 'annotations/val.json',
      'help': 'COCO JSON annotations describing the validation images and bounding boxes.',
      'label': 'Validation annotations file',
      'default_hint': 'Default: annotations/val.json.'},
     {'name': 'epochs',
      'flag': '--epochs',
      'kind': 'int',
      'help': 'Maximum number of passes through the training dataset.',
      'label': 'Training epochs',
      'default': '150',
      'default_hint': 'Default: 150.'},
     {'name': 'batch_size',
      'flag': '--batch-size',
      'kind': 'int',
      'help': 'Images processed per training batch. Larger values require more memory.',
      'label': 'Batch size',
      'default': '1',
      'default_hint': 'Default: 1.'},
     {'name': 'num_workers',
      'flag': '--num-workers',
      'kind': 'int',
      'help': 'Number of processes loading images. Set 0 if multiprocessing causes problems.',
      'label': 'Data loading workers (optional)',
      'default_hint': 'Automatic based on RAM and CPU; 0 on macOS.'},
     {'name': 'val_frequency',
      'flag': '--val-frequency',
      'kind': 'int',
      'default': '1',
      'help': 'Evaluate the model after this many epochs. 1 means every epoch.',
      'label': 'Validation interval (epochs)',
      'default_hint': 'Default: 1.'},
     {'name': 'early_stopping_patience',
      'flag': '--early-stopping-patience',
      'kind': 'int',
      'help': 'Stop after this many validation checks without improvement.',
      'label': 'Early stopping patience (optional)',
      'default_hint': 'Disabled. Leave blank to run up to the maximum epochs.'},
     {'name': 'lr',
      'flag': '--lr',
      'kind': 'float',
      'default': '0.05',
      'help': 'Step size for the SGD optimizer.',
      'label': 'Learning rate',
      'default_hint': 'Default: 0.05.'},
     {'name': 'no_amp',
      'flag': '--no-amp',
      'kind': 'checkbox',
      'help': 'Turn off CUDA mixed precision. Usually leave unchecked.',
      'label': 'Disable mixed precision',
      'default_hint': 'Unchecked; mixed precision is used on CUDA.'},
     {'name': 'no_pin_memory',
      'flag': '--no-pin-memory',
      'kind': 'checkbox',
      'help': 'Turn off pinned memory for data loading. Usually leave unchecked.',
      'label': 'Disable pinned memory',
      'default_hint': 'Unchecked; pinned memory is used on CUDA.'},
     {'name': 'clear_cache_each_step',
      'flag': '--clear-cache-each-step',
      'kind': 'checkbox',
      'help': 'Can help diagnose memory issues but slows training.',
      'label': 'Clear GPU cache after each step',
      'default_hint': 'Unchecked.'},
     {'name': 'best_metric',
      'flag': '--best-metric',
      'kind': 'select',
      'choices': ['ap50', 'map50_95'],
      'default': 'ap50',
      'help': 'ap50 measures average precision at IoU 0.5; map50_95 averages across IoU thresholds from 0.5 to '
              '0.95.',
      'label': 'Best model selection metric',
      'default_hint': 'Default: ap50.'}]


def _test_fields():
    return [
        {"name": "data_dir", "picker": "folder", "flag": "--data-dir", "kind": "text", "default": "data/test", "help": "Directory containing test tiles."},
        {"name": "annotation_file", "picker": "file", "flag": "--annotation-file", "kind": "text", "default": "annotations/test.json", "help": "COCO annotation JSON for the test split."},
        {"name": "checkpoint", "picker": "file", "flag": "--checkpoint", "kind": "text", "default": "checkpoints/best_model.pt", "help": "Model checkpoint to evaluate."},
        {"name": "results_dir", "picker": "folder", "flag": "--results-dir", "kind": "text", "default": "paper_results", "help": "Directory for output metrics and plots."},
    ]


def _project_boxes_fields():
    return [
        {"name": "dataset_path", "picker": "folder", "flag": "--dataset-path", "kind": "text", "help": "ODM project folder containing opensfm/, odm_dem/, and odm_georeferencing/."},
        {"name": "predictions_json", "picker": "file", "flag": "--predictions-json", "kind": "text", "help": "Module 7 prediction JSON (full COCO dict or flat COCO prediction list)."},
        {"name": "annotation_json", "picker": "file", "flag": "--annotation-json", "kind": "text", "help": "Optional Module 7 COCO test annotation JSON for image_id to filename mapping."},
        {"name": "image_list", "picker": "file", "flag": "--image-list", "kind": "text", "help": "Optional text file with one image filename per line, matching image_id order."},
        {"name": "tile_metadata", "picker": "file", "flag": "--tile-metadata", "kind": "text", "help": "Optional tile metadata JSON from Module 5."},
        {"name": "output_dir", "picker": "folder", "flag": "--output-dir", "kind": "text", "help": "Output folder name, created inside dataset path unless absolute."},
        {"name": "dem_filename", "picker": "file", "flag": "--dem-filename", "kind": "text", "help": "DSM path relative to dataset path."},
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
        "kind": "cli",
        "script": {"any": "gps_embed.py"},
        "fields": [
            {"name": "utm_zone", "label": "UTM zone", "flag": "--utm-zone", "kind": "int", "required": True, "min": 1, "max": 60, "help": "Enter the zone for your event coordinates, from 1 to 60."},
            {"name": "hemisphere", "label": "Hemisphere", "flag": "--hemisphere", "kind": "select", "choices": ["N", "S"], "required": True, "help": "N for Northern Hemisphere; S for Southern Hemisphere."},
            {"name": "images_folder", "picker": "folder", "label": "Images folder", "flag": "--images-folder", "kind": "text", "required": True, "help": "Full path to the folder containing JPG images and events.txt. Outputs are saved in this folder."},
        ],
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
        "script": {"any": "run_ODM_process.py"},
        "fields": [
            {"name": "project_folder", "picker": "folder", "required": True, "kind": "cwd", "label": "ODM project folder", "help": "Folder containing an images/ subfolder (and optional images/gcp_list.txt). The script is run with this as its working directory."},
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

# User-facing labels describe the workflow; kind still controls execution.
MODULE_TYPES = {
    "desktop": {"label": "Desktop app", "description": "Opens a separate desktop window on this computer."},
    "form": {"label": "Form", "description": "Enter inputs on the page, then run and follow the live log."},
    "training": {"label": "Training", "description": "Train or fine-tune a model using a settings form and live log. Transfer Learning also offers advanced YAML settings."},
    "docker": {"label": "Docker", "description": "Processes a project through Docker, launched by Python. Requires Docker Desktop running."},
}
for module in MODULES:
    if module["id"] in ("6-1_obj_det_trainer", "6-2_obj_det_trans_learner"):
        module["display_kind"] = "training"
    else:
        module["display_kind"] = {"gui": "desktop", "cli": "form", "shell": "docker"}[module["kind"]]
    module["type_label"] = MODULE_TYPES[module["display_kind"]]["label"]

# Shared field guidance and validation rules for every command-line form.
for module in MODULES:
    for field in module['fields']:
        field.setdefault('label', field['name'].replace('_', ' ').capitalize())
        if field.get('default') is not None:
            field.setdefault('default_hint', f"Default: {field['default']}.")
        if field['kind'] == 'int' and field['name'] != 'seed':
            field.setdefault('min', 0 if field['name'] == 'num_workers' else 1)
        if field['kind'] == 'float':
            field['positive'] = True
        if field['name'] in ('output_dir', 'results_dir'):
            field['path_role'] = 'output'

projection = next(module for module in MODULES if module['id'] == '8_img_to_ortho')
for field in projection['fields']:
    if field['name'] in ('dataset_path', 'predictions_json'):
        field['required'] = True
    if field['name'] in ('output_dir', 'dem_filename'):
        field['path_base'] = 'dataset'
    defaults = {'output_dir': 'orthorectified2', 'dem_filename': 'odm_dem/dsm.tif', 'num_threads': '1'}
    if field['name'] in defaults:
        field['default'] = defaults[field['name']]
        field['default_hint'] = f"Default: {field['default']}."
    if field['name'] == 'num_threads':
        field['label'] = 'Worker threads'
        field['help'] = 'Parallel pixel-processing threads. Start with 1; increase only if it improves performance.'
    if field['name'] in ('annotation_json', 'image_list', 'tile_metadata'):
        field['label'] += ' (optional)'
    if field['name'] == 'annotation_json':
        field['help'] = 'Use the test COCO annotations from Module 7 to map prediction image IDs to filenames. Required for a flat prediction list unless an image list is provided.'

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
        display_variant = {"mac": "macOS", "win": "Windows"}.get(variant, variant)
        label = "Script" if variant == "any" else f"Script ({display_variant})"
        links.append((label, github_url(module, filename)))
    if req_name:
        links.append((req_name, github_url(module, req_name)))
    if module["kind"] == "config":
        links.append((module["config_file"], github_url(module, module["config_file"])))
    return links
