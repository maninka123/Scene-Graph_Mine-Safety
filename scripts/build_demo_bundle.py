from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.demo_service import load_demo


def main() -> None:
    destination = ROOT / "web" / "public" / "demo.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(load_demo(), separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {destination} ({destination.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
