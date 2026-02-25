from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


def save_checkpoint(path: str | Path, state: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str = "cpu",
) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    try:
        state = torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only.
        state = torch.load(p, map_location=map_location)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    return state
