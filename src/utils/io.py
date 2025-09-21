import os, json
from pathlib import Path

def ensure_dir(p: str):
    # create dir if not exists
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path) as f:
        return json.load(f)
