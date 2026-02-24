from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path_value: str | Path, root: Path | None = None) -> Path:
    root = root or repo_root()
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (root / path_obj).resolve()


def resolve_existing_dir(candidates: Iterable[str | Path], root: Path | None = None) -> Path:
    root = root or repo_root()
    for candidate in candidates:
        resolved = resolve_path(candidate, root=root)
        if resolved.exists() and resolved.is_dir():
            return resolved
    joined = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"None of the candidate directories exist: {joined}")


def list_pcd_files(folder: str | Path) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return sorted(root.glob("*.pcd"))


def write_json(path: str | Path, payload: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

