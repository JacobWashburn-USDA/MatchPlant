"""Transfer-learning form schema and validated YAML updates."""
import copy
import json
from pathlib import Path
import yaml

FIELDS = [
    ("pretrained_model", "Pretrained model file", ("transfer", "pretrained_model_path"), "file", "best_model.pt", "Model checkpoint to fine-tune. Select the .pt file from your earlier training."),
    ("train_dir", "Training images folder", ("data", "train_dir"), "folder", "data/new_train", "Folder containing image tiles for the new training dataset."),
    ("val_dir", "Validation images folder", ("data", "val_dir"), "folder", "data/new_val", "Folder containing images used to evaluate fine-tuning progress."),
    ("train_annotations", "Training annotations file", ("data", "train_annotations"), "file", "annotations/new_train.json", "COCO JSON annotations for the new training images."),
    ("val_annotations", "Validation annotations file", ("data", "val_annotations"), "file", "annotations/new_val.json", "COCO JSON annotations for the validation images."),
    ("checkpoint_dir", "Save checkpoints to folder", ("training", "checkpoint_dir"), "folder", "checkpoints", "Choose where to save the fine-tuned model. Type a new folder path for each experiment, or browse for an existing folder."),
    ("validation_dir", "Save validation reports to folder", ("training", "validation_dir"), "folder", "validation_results", "Folder for per-epoch validation JSON files and the training summary. A new folder is created when needed."),
    ("epochs", "Training epochs", ("transfer", "training", "epochs"), "int", 50, "Maximum number of passes through the new training dataset."),
    ("batch_size", "Batch size", ("transfer", "training", "batch_size"), "int", 1, "Images per training batch. Larger values need more memory."),
    ("initial_epoch", "Starting epoch", ("transfer", "training", "initial_epoch"), "int", 0, "Starting epoch index. Keep 0 for a new fine-tuning run; this alone does not restore optimizer state."),
    ("freeze_backbone", "Freeze backbone", ("transfer", "freeze_backbone"), "checkbox", True, "Keep the feature-extraction backbone weights unchanged."),
    ("freeze_rpn", "Freeze region proposal network", ("transfer", "freeze_rpn"), "checkbox", False, "Keep the network that proposes candidate object regions unchanged."),
    ("freeze_roi_heads", "Freeze detection heads", ("transfer", "freeze_roi_heads"), "checkbox", False, "Keep the final classification and bounding-box prediction layers unchanged."),
    ("lr_backbone", "Backbone learning rate", ("transfer", "learning_rates", "backbone"), "float", 0.0001, "Used only when the backbone is not frozen."),
    ("lr_rpn", "Region proposal learning rate", ("transfer", "learning_rates", "rpn"), "float", 0.001, "Used only when the region proposal network is not frozen."),
    ("lr_roi_heads", "Detection heads learning rate", ("transfer", "learning_rates", "roi_heads"), "float", 0.001, "Used only when the detection heads are not frozen."),
]


def load(text):
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("The YAML configuration is invalid. Correct it under Advanced YAML settings.") from exc
    if not isinstance(data, dict):
        raise ValueError("The YAML configuration must contain named sections.")
    return data


def get(data, path, default):
    current = data
    for key in path:
        if not isinstance(current, dict):
            raise ValueError("Configuration section must be a mapping: " + ".".join(path))
        if key not in current:
            return default
        current = current[key]
    return current


def fields(text):
    data = load(text)
    result = []
    for name, label, path, kind, default, help_text in FIELDS:
        field = dict(name=name, label=label, kind="text" if kind in ("file", "folder") else kind,
                     value=get(data, path, default), default=str(default), help=help_text,
                     default_hint="Default: " + ("checked" if default is True else "unchecked" if default is False else str(default)) + ".")
        if kind in ("folder", "file"):
            field.update(picker=kind, required=name not in ("checkpoint_dir", "validation_dir"))
            if name in ("checkpoint_dir", "validation_dir"):
                field["path_role"] = "output"
        elif kind != "checkbox":
            field.update(required=True, min=0 if name == "initial_epoch" else 1 if kind == "int" else 0)
        result.append(field)
    return result


def updated(text, form):
    import math
    data = copy.deepcopy(load(text))
    values = {}
    for name, label, path, kind, default, help_text in FIELDS:
        raw = form.get(name, "")
        if name in ("checkpoint_dir", "validation_dir") and not raw.strip():
            raw = str(get(data, path, default))
        if kind == "checkbox":
            value = raw == "on"
        elif kind in ("int", "float"):
            try:
                value = int(raw) if kind == "int" else float(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(label + " must be a number.") from exc
            if not math.isfinite(value) or value < 0 or (name != "initial_epoch" and value == 0):
                raise ValueError(label + " must be " + ("zero or greater." if name == "initial_epoch" else "greater than zero."))
        else:
            value = raw.strip()
            if not value:
                raise ValueError("Select " + label.lower() + ".")
        values[path] = value
        section = data
        for key in path[:-1]:
            section = section.setdefault(key, {})
            if not isinstance(section, dict):
                raise ValueError("Invalid configuration section: " + key)
        section[path[-1]] = value
    if all(values[("transfer", key)] for key in ("freeze_backbone", "freeze_rpn", "freeze_roi_heads")):
        raise ValueError("Unfreeze at least one model component so there are weights to train.")
    # Replace existing scalar nodes in place to retain comments and advanced settings.
    tree = yaml.compose(text)
    edits = []
    for path, value in values.items():
        node = tree
        for key in path:
            if not isinstance(node, yaml.MappingNode):
                break
            node = next((v for k, v in node.value if k.value == key), None)
            if node is None:
                break
        if not isinstance(node, yaml.ScalarNode):
            return yaml.safe_dump(data, sort_keys=False)
        scalar = json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value).lower()
        edits.append((node.start_mark.index, node.end_mark.index, scalar))
    for start, end, scalar in sorted(edits, reverse=True):
        text = text[:start] + scalar + text[end:]
    load(text)
    return text


def save(path, form):
    path = Path(path)
    text = updated(path.read_text(encoding="utf-8"), form)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
