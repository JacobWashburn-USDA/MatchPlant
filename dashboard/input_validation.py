"""Validate dashboard values before starting a module process."""
import math
from pathlib import Path


def validated_form(module, form, directory):
    result = dict(form)
    directory = Path(directory)
    for field in module['fields']:
        name = field['name']
        label = field.get('label', name.replace('_', ' ').capitalize())
        value = str(form.get(name) or field.get('default') or '').strip()
        if not value:
            if field.get('required'):
                raise ValueError(f"Choose or enter {label.lower()}.")
            continue
        if field['kind'] in ('int', 'float'):
            try:
                number = int(value) if field['kind'] == 'int' else float(value)
            except ValueError:
                raise ValueError(f"{label} must be a {'whole number' if field['kind'] == 'int' else 'number'}.")
            if not math.isfinite(number):
                raise ValueError(f"{label} must be finite.")
            if 'min' in field and number < field['min']:
                raise ValueError(f"{label} must be at least {field['min']}.")
            if field.get('positive') and number <= 0:
                raise ValueError(f"{label} must be greater than zero.")
            if 'max' in field and number > field['max']:
                raise ValueError(f"{label} must be at most {field['max']}.")
        if field['kind'] == 'select' and value not in field['choices']:
            raise ValueError(f"Choose a valid option for {label.lower()}.")
        if field.get('picker'):
            path = Path(value).expanduser()
            base = directory
            if field.get('path_base') == 'dataset':
                dataset = Path(str(form.get('dataset_path') or '')).expanduser()
                base = dataset if dataset.is_absolute() else directory / dataset
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            if field.get('path_role') == 'output':
                if path.exists() and not path.is_dir():
                    raise ValueError(f"{label} must be a folder, not a file.")
            else:
                valid = path.is_file() if field['picker'] == 'file' else path.is_dir()
                if not valid:
                    raise ValueError(f"{label} was not found. Browse for an existing {field['picker']}.")
            result[name] = str(path)
        elif field['kind'] != 'checkbox':
            result[name] = value
    return result
