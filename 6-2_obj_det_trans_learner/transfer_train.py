"""Fine-tune a Faster R-CNN checkpoint using transfer_config.yaml."""
import argparse
import copy
import importlib.util
import math
import os
from pathlib import Path
import sys
import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain named sections.")
    for section in ("transfer", "model", "data"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid {section} section.")
    transfer = config["transfer"]
    training = transfer.get("training", {})
    for name in ("epochs", "batch_size"):
        if type(training.get(name)) is not int or training[name] < 1:
            raise ValueError(f"transfer.training.{name} must be a positive integer.")
    initial = training.get("initial_epoch", 0)
    if type(initial) is not int or not 0 <= initial < training["epochs"]:
        raise ValueError("Starting epoch must be between zero and epochs minus one.")
    for name in ("freeze_backbone", "freeze_rpn", "freeze_roi_heads"):
        if type(transfer.get(name)) is not bool:
            raise ValueError(f"transfer.{name} must be true or false.")
    if all(transfer[name] for name in ("freeze_backbone", "freeze_rpn", "freeze_roi_heads")):
        raise ValueError("Unfreeze at least one component to train.")
    for name in ("backbone", "rpn", "roi_heads"):
        value = transfer.get("learning_rates", {}).get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Learning rate for {name} must be positive and finite.")
    if type(config["model"].get("num_classes")) is not int or config["model"]["num_classes"] < 2:
        raise ValueError("num_classes must include background and at least one object class.")
    workers = config["data"].get("num_workers", 0)
    if type(workers) is not int or workers < 0:
        raise ValueError("num_workers must be zero or greater.")
    for name in ("pin_memory", "persistent_workers"):
        if name in config["data"] and type(config["data"][name]) is not bool:
            raise ValueError(f"data.{name} must be true or false.")
    save_freq = config.get("training", {}).get("save_freq", 5)
    if type(save_freq) is not int or save_freq < 1:
        raise ValueError("training.save_freq must be positive.")
    def resolve(value):
        value = Path(value).expanduser()
        return value.resolve() if value.is_absolute() else (path.parent / value).resolve()
    checkpoint = resolve(transfer.get("pretrained_model_path", ""))
    if not checkpoint.is_file():
        raise ValueError(f"Pretrained model file not found: {checkpoint}")
    transfer["pretrained_model_path"] = str(checkpoint)
    for name in ("train_dir", "val_dir", "train_annotations", "val_annotations"):
        value = config["data"].get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Set data.{name}.")
        resolved = resolve(value)
        valid = resolved.is_dir() if name.endswith("_dir") else resolved.is_file()
        if not valid:
            raise ValueError(f"Dataset path not found: {resolved}")
        config["data"][name] = str(resolved)
    output = config.setdefault("training", {})
    output["checkpoint_dir"] = str(resolve(output.get("checkpoint_dir", "checkpoints")))
    output["validation_dir"] = str(resolve(output.get("validation_dir", "validation_results")))
    return config


def base_training():
    path = Path(__file__).resolve().parents[1] / "6-1_obj_det_trainer" / "train.py"
    if not path.is_file():
        raise ValueError("Keep Module 6.1 beside Module 6.2 in the MatchPlant repository.")
    spec = importlib.util.spec_from_file_location("matchplant_base_training", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_transfer_model(config, state):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
    options = copy.deepcopy(config)
    architecture = options.pop("architecture", "auto")
    if architecture == "auto":
        architecture = "v1" if "roi_heads.box_head.fc6.weight" in state else "v2"
    if architecture not in ("v1", "v2"):
        raise ValueError("model.architecture must be auto, v1, or v2.")
    sizes = tuple(tuple(group) for group in options.pop("anchor_sizes"))
    ratios = tuple(tuple(group) for group in options.pop("anchor_ratios"))
    anchor_source = options.pop("anchor_source", "auto")
    if anchor_source not in ("auto", "config"):
        raise ValueError("model.anchor_source must be auto or config.")
    expected = state.get("rpn.head.cls_logits.weight")
    count = int(expected.shape[0]) if expected is not None else None
    configured_count = len(sizes[0]) * len(ratios[0])
    if count is not None and count != configured_count:
        known = {3: (((32,), (64,), (128,), (256,), (512,)), ((0.5, 1.0, 2.0),) * 5),
                 12: (((32, 64, 96, 128),), ((0.5, 1.0, 2.0),)),
                 40: (((4, 8, 16, 32, 64, 96, 128, 256),), ((0.5, 0.75, 1.0, 1.5, 2.0),))}
        if anchor_source == "auto" and count in known:
            sizes, ratios = known[count]
            print(f"Using known MatchPlant anchor layout for checkpoint ({count} anchors).", flush=True)
        else:
            raise ValueError(f"Checkpoint has {count} anchors; configured model has {configured_count}. Set matching anchor_sizes/anchor_ratios.")
    if len(sizes) == 1 and len(ratios) == 1:
        sizes, ratios = sizes * 5, ratios * 5
    if len(sizes) != 5 or len(ratios) != 5:
        raise ValueError("ResNet FPN needs one shared anchor group or five feature-level groups.")
    constructor = fasterrcnn_resnet50_fpn if architecture == "v1" else fasterrcnn_resnet50_fpn_v2
    generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    counts = generator.num_anchors_per_location()
    if len(set(counts)) != 1:
        raise ValueError("All FPN levels must have the same number of anchors.")
    model = constructor(weights=None, weights_backbone=None, **options)
    model.rpn.anchor_generator = generator
    model.rpn.head = RPNHead(model.backbone.out_channels, counts[0], conv_depth=2 if architecture == "v2" else 1)
    config.update(architecture=architecture, anchor_source="config",
                  anchor_sizes=[list(group) for group in sizes], anchor_ratios=[list(group) for group in ratios])
    print(f"Model architecture: {architecture}; no pretrained weights download needed.", flush=True)
    return model


def train_epoch(model, optimizer, loader, device, transfer):
    import torch
    model.train()
    if transfer["freeze_backbone"]:
        model.backbone.eval()
    total = 0.0
    for step, (images, targets) in enumerate(loader, 1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        losses = sum(model(images, targets).values())
        if not torch.isfinite(losses):
            raise ValueError("Training loss is not finite.")
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += losses.item()
        print(f"Batch {step}/{len(loader)} loss={losses.item():.4f}", flush=True)
    return total / len(loader)


def run(config):
    import torch
    from torch.utils.data import DataLoader
    from transfer_utils import read_checkpoint, load_pretrained_model, freeze_layers, get_transfer_optimizer, print_trainable_parameters
    resources = config.get("resources", {})
    if resources.get("memory_config"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", resources["memory_config"])
    torch.backends.cudnn.benchmark = bool(resources.get("cuda_benchmark", True))
    base = base_training()
    if base.COCO is None:
        raise ValueError("Install pycocotools before training.")
    device, _ = base.configure_device_and_resources()
    transfer, data = config["transfer"], config["data"]
    checkpoint, state = read_checkpoint(transfer["pretrained_model_path"])
    model = build_transfer_model(config["model"], state)
    load_pretrained_model(model, transfer["pretrained_model_path"], checkpoint=checkpoint)
    freeze_layers(model, transfer)
    model.to(device)
    print_trainable_parameters(model)
    loaders = []
    for training, directory, annotations in [(True, "train_dir", "train_annotations"), (False, "val_dir", "val_annotations")]:
        dataset = base.MaizeDatasetCOCO(Path(data[directory]), Path(data[annotations]), transforms=base.get_transform(train=training))
        if not len(dataset):
            raise ValueError(f"Dataset contains no annotated images: {data[directory]}")
        loaders.append(DataLoader(dataset, batch_size=transfer["training"]["batch_size"], shuffle=training,
                       num_workers=data.get("num_workers", 0), collate_fn=base.collate_fn,
                       pin_memory=data.get("pin_memory", False) and device.type == "cuda",
                       persistent_workers=data.get("persistent_workers", False) and data.get("num_workers", 0) > 0))
    optimizer = get_transfer_optimizer(model, transfer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    output = Path(config["training"]["checkpoint_dir"])
    validation = Path(config["training"]["validation_dir"])
    output.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    if (output / "best_model_transfer.pt").exists():
        raise ValueError("Output already contains best_model_transfer.pt. Choose a new checkpoint folder in the form or training.checkpoint_dir in YAML.")
    best = -math.inf
    try:
        for epoch in range(transfer["training"].get("initial_epoch", 0), transfer["training"]["epochs"]):
            print(f"Epoch {epoch + 1}/{transfer['training']['epochs']}", flush=True)
            loss = train_epoch(model, optimizer, loaders[0], device, transfer)
            stats = base.evaluate(model, loaders[1], device, epoch=epoch, train_loss=loss, results_dir=validation)
            ap50 = float(stats[1])
            if not math.isfinite(ap50) or ap50 < 0:
                raise ValueError("Validation did not produce a valid AP50 score.")
            scheduler.step()
            payload = dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                           scheduler_state_dict=scheduler.state_dict(), AP_50=ap50, mAP_50_95=float(stats[0]),
                           train_loss=loss, model_config=config["model"])
            if ap50 > best:
                best = ap50
                payload["best_ap50"] = best
                torch.save(payload, output / "best_model_transfer.pt")
                print(f"Saved best transfer model: AP50={ap50:.4f}", flush=True)
            if epoch % config["training"].get("save_freq", 5) == 0:
                torch.save(payload, output / f"model_transfer_epoch_{epoch}.pt")
        print("Transfer learning complete.", flush=True)
    except Exception:
        torch.save(model.state_dict(), output / "emergency_save_transfer.pt")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("transfer_config.yaml")))
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration and paths without training.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.validate_only:
            print("Configuration and paths validated.")
        else:
            run(config)
    except (ValueError, OSError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        print(f"[transfer learning] {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

