import yaml
import importlib

def load_yaml(file):
    with open(file,'rt') as f:
        out = yaml.safe_load(f)
    return out

def get_object_from_path(p):
    parts = p.split(".")
    path, name = ".".join(parts[:-1]), parts[-1]
    pkg = importlib.import_module(path)
    obj = getattr(pkg,name)
    return obj