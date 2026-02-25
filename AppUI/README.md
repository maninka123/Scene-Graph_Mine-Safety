# AppUI: Interactive Segmentation + Scene Graph + LLM Dashboard

This app is a separate web interface for running your full pipeline without touching the existing scripts.

- Single-frame mode: `MinkUNET segmentation -> scene graph -> LLM`
- Temporal mode: selected frame sequence with timestamp range + gap control
- Outputs are saved only under: `AppUI/app_results/`

## Install

From project root:

```bash
pip install -r requirements.txt
pip install -r AppUI/requirements.txt
```

## Run

From project root:

```bash
streamlit run AppUI/app.py
```

## Key Features

1. Select a single `.pcd` frame or a folder of `PC_*.pcd` frames.
2. Temporal frame filtering:
   - Index range selection
   - Minimum time gap slider (seconds)
3. Hyperparameter controls:
   - MinkUNET config/checkpoint
   - Temporal smoothing window
   - Graph thresholds
   - LLM model toggle/name
   - Optional class-wise DBSCAN overrides
4. Live progress and live segmentation preview during temporal runs.
5. Saved analytics:
   - 3D semantic preview
   - class distribution
   - temporal inference/object trends
   - movement-state summary
6. LLM model catalog and runtime guidance:
   - auto-detects local models in `LLM/Model_Cache` and `LLM/DeepSeek_Model`
   - lets user select detected model or enter custom model id
   - shows device-aware compatibility guidance (`can run`, `can run but not ideal`, `likely not suitable`)
7. Structured LLM report view with emoji categories:
   - safety, operations, human-related, infrastructure, movement, and general observations
8. All generated plots are saved as HTML under each run folder.

## Output Layout

```text
AppUI/app_results/
  single/<run_name_timestamp>/
    <frame_name>/
      scene_objects_segmented.json
      semantic_segmentation.pcd
      semantic_prediction.npz
      scene_graph.json
      llm_prompt.txt
      LLM/...
      plots/...
    run_summary.json

  temporal/<run_name_timestamp>/
    frames/<frame_name>/
      scene_objects_segmented.json
      semantic_segmentation.pcd
      semantic_prediction.npz
      summary.json
    frame_metrics.csv
    temporal_scene_graph.json
    llm_temporal_prompt.txt
    LLM/...
    plots/...
    run_summary.json
```
