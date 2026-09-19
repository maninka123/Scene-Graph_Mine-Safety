from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mine_safety.config import load_config
from mine_safety.perception.inference import PointCloudPerceiver
from mine_safety.pipeline import MineSafetyPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Nine-class semantic checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "assessment.json")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    config = load_config()
    perception = PointCloudPerceiver(config, args.checkpoint).predict(args.point_cloud)
    result = MineSafetyPipeline(config).process_nodes(
        perception["nodes"], graph_id=args.point_cloud.stem,
        mean_intensity=perception["mean_intensity"], run_llm=args.llm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
