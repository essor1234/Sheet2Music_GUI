from pathlib import Path

def create_paths(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"{path} created")