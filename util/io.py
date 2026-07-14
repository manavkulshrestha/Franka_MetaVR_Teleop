import yaml

def yaml_to_dict(path: str) -> dict:
    with open(path, 'r') as file:
        try:
            data_dict = yaml.safe_load(file)
            return data_dict
        except yaml.YAMLError as e:
            print(f'Error parsing YAML file: {e}')