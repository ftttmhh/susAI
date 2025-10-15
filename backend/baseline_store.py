import json
import threading
from pathlib import Path
from typing import Optional

_LOCK = threading.Lock()
_PATH = Path(__file__).with_name("go_to_models.json")


def _read_store() -> dict:
    if not _PATH.exists():
        return {}
    try:
        with _PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_store(d: dict):
    with _PATH.open("w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def get_baseline(task: str) -> Optional[str]:
    if not task:
        return None
    with _LOCK:
        store = _read_store()
        return store.get(str(task))


def set_baseline(task: str, model: str):
    if not task or not model:
        return
    with _LOCK:
        store = _read_store()
        store[str(task)] = str(model)
        _write_store(store)
