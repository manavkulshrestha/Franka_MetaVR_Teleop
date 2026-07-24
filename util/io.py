import yaml
from pathlib import Path
from itertools import count


def yaml_to_dict(path: str) -> dict:
    with open(path, 'r') as file:
        try:
            data_dict = yaml.safe_load(file)
            return data_dict
        except yaml.YAMLError as e:
            print(f'Error parsing YAML file: {e}')


def versioned_dir(parent: Path) -> Path:
    try:
        used = {
            int(item.name)
            for item in parent.iterdir()
            if item.is_dir() and item.name.isdigit()
        }
    except FileNotFoundError:
        return parent/'0'

    smallest_available = next(n for n in count(0) if n not in used)
    return parent/str(smallest_available)