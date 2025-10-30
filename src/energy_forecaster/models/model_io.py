
from pathlib import Path
from typing import Any, Optional, Dict
from joblib import dump, load
import json

def save_model(model: Any, path: str | Path, meta: Optional[Dict] = None, compress: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path, compress=compress)
    if meta is not None:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def load_model(path: str | Path):
    return load(path)
