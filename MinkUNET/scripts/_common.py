from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minkunet.config import load_config  # noqa: E402
from minkunet.utils.io import resolve_path  # noqa: E402


def load_cfg(config_path: str) -> Dict:
    cfg = load_config(config_path)
    return cfg


def abs_path(path_like: str | Path) -> Path:
    return resolve_path(path_like, root=REPO_ROOT)

