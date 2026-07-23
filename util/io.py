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


def available_version(parent: Path) -> int:
    used = {
        int(item.name)
        for item in parent.iterdir()
        if item.is_dir() and item.name.isdigit()
    }

    smallest_available = next(n for n in count(0) if n not in used)
    return smallest_available