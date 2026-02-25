from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


PCD_PATTERN = re.compile(r"PC_(\d{8})_(\d{6})(\d{0,6})\.pcd$", re.IGNORECASE)


def parse_pcd_timestamp(path: Path) -> Optional[datetime]:
    match = PCD_PATTERN.search(path.name)
    if not match:
        return None

    date_str, hms_str, frac_str = match.groups()
    try:
        base = datetime.strptime(f"{date_str}{hms_str}", "%Y%m%d%H%M%S")
    except ValueError:
        return None

    if frac_str:
        # Keep microsecond precision (pad/truncate to 6 digits).
        usec = int(frac_str[:6].ljust(6, "0"))
        base = base.replace(microsecond=usec)
    return base


def list_pcd_files(folder: str) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        return []

    files = [p for p in root.glob("PC_*.pcd") if p.is_file()]
    files.sort(key=lambda p: (parse_pcd_timestamp(p) is None, parse_pcd_timestamp(p), p.name))
    return files


def frame_rows(files: List[Path]) -> List[dict]:
    rows = []
    for idx, pcd_path in enumerate(files):
        ts = parse_pcd_timestamp(pcd_path)
        rows.append(
            {
                "index": idx,
                "file_name": pcd_path.name,
                "timestamp": ts.isoformat() if ts else "N/A",
                "path": str(pcd_path.resolve()),
            }
        )
    return rows


def select_index_range(files: List[Path], start_idx: int, end_idx: int) -> List[Path]:
    if not files:
        return []
    start = max(0, int(start_idx))
    end = min(len(files) - 1, int(end_idx))
    if end < start:
        return []
    return files[start : end + 1]


def select_with_gap_seconds(files: List[Path], gap_seconds: int) -> List[Path]:
    if not files:
        return []
    gap = max(1, int(gap_seconds))

    selected: List[Path] = []
    last_ts: Optional[datetime] = None
    for pcd_path in files:
        ts = parse_pcd_timestamp(pcd_path)
        if not selected:
            selected.append(pcd_path)
            last_ts = ts
            continue

        if ts is None or last_ts is None:
            # If no timestamp is available, keep every file in the already-range-filtered set.
            selected.append(pcd_path)
            last_ts = ts
            continue

        if (ts - last_ts).total_seconds() >= gap:
            selected.append(pcd_path)
            last_ts = ts
    return selected


def range_and_gap_select(
    files: List[Path], index_range: Tuple[int, int], gap_seconds: int
) -> List[Path]:
    ranged = select_index_range(files, start_idx=index_range[0], end_idx=index_range[1])
    return select_with_gap_seconds(ranged, gap_seconds=gap_seconds)
