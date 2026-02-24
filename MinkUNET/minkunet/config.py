from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_update(output[key], value)
        else:
            output[key] = value
    return output


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(config_path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = load_yaml(cfg_path)
    base_cfg_path = cfg.get("base_config")
    if base_cfg_path:
        candidates = [Path(base_cfg_path), cfg_path.parent / Path(base_cfg_path)]
        resolved_base = None
        for candidate in candidates:
            if candidate.exists():
                resolved_base = candidate
                break
        if resolved_base is None:
            raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
        base_cfg = load_yaml(resolved_base)
        cfg = _deep_update(base_cfg, {k: v for k, v in cfg.items() if k != "base_config"})
    return cfg
