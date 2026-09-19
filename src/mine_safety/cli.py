from __future__ import annotations

import argparse
import json
from pathlib import Path

from mine_safety.pipeline import MineSafetyPipeline
from mine_safety.schemas import SceneGraph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the graph-to-safety portion of the paper pipeline.")
    parser.add_argument("scene_graph", type=Path, help="Input scene graph JSON")
    parser.add_argument("--output", type=Path, default=Path("outputs/assessment.json"))
    parser.add_argument("--llm", action="store_true", help="Enable local Qwen2.5-3B reasoning")
    args = parser.parse_args()
    graph = SceneGraph.model_validate_json(args.scene_graph.read_text(encoding="utf-8"))
    result = MineSafetyPipeline().process_graph(graph, run_llm=args.llm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
