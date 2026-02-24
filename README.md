# 3D Mine Scene Understanding: Scene Graphs with Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-Point_Cloud-green)](http://www.open3d.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MinkowskiEngine](https://img.shields.io/badge/MinkowskiEngine-Sparse_3D-orange)](https://github.com/NVIDIA/MinkowskiEngine)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![Scene%20Graph](https://img.shields.io/badge/Scene%20Graph-Spatial%20Reasoning-6f42c1)](#a-scene-graph-pipeline-main-logic)
[![Segmentation](https://img.shields.io/badge/Segmentation-MinkUNET-0ea5e9)](#b-minkunet-segmentation-pipeline)

This repository follows a two-part logic for underground mining point clouds:

1. Segmentation stage: produce object/semantic partitions per frame.
2. Scene graph stage: use those segmented objects to build spatial and temporal scene graphs, then run LLM reasoning.

Scene graphs are the main reasoning layer. Segmentation is the upstream stage that provides object regions/labels used by graph construction.

## 📁 Project Layout

```text
Scene-Graph-Mine-Safety/
├── [Main] run_pipeline.py                 # 🚀 Single-frame pipeline (segment -> graph -> LLM -> visualize)
├── [Main] run_temporal_pipeline.py        # 🚀 Temporal pipeline (tracking + temporal graph + LLM)
│
├── func_segment_clustering.py             # Single-frame geometric segmentation (DBSCAN)
├── func_segment_clustering_temporal.py    # Multi-frame segmentation across dataset sequence
├── func_build_graph.py                    # Build single-frame scene graph (nodes + near-edges)
├── func_build_temporal_graph.py           # Track objects over time + temporal graph construction
├── func_query_local_llm.py                # Local LLM inference for single-frame graph prompt
├── func_query_llm_temporal.py             # Local LLM inference for temporal graph prompt
├── func_visualize_scene_graph.py          # Visualize single-frame graph and labels
├── func_visualize_temporal.py             # Visualize trajectories and temporal context
│
├── MinkUNET/
│   ├── configs/                           # Training/inference configs
│   ├── scripts/                           # CLI scripts (train, validate, infer, ablations)
│   ├── minkunet/                          # Core model/data/engine code
│   ├── checkpoints/                       # Saved model checkpoints
│   ├── logs/                              # Metrics, plots, diagnostics
│   └── data/                              # Manifests and metadata
│
├── LLM/
│   └── Model_Cache/                       # 🤗 Downloaded local model cache
├── Datasets/                              # Input point cloud datasets
├── Results/                               # Output artifacts from runs
└── requirements.txt                       # Python dependencies
```

## A) Scene Graph Pipeline (Main Logic)

### Two modes

- Single-frame graph reasoning:
  - `python "[Main] run_pipeline.py"`
- Temporal graph reasoning (multi-frame tracking):
  - `python "[Main] run_temporal_pipeline.py"`

### Stage flow (single-frame)

1. `func_segment_clustering.py`
   - voxel downsample, RANSAC plane removal, DBSCAN clustering
   - writes `scene_objects_clustered.json` and `colored_clusters.pcd`
2. `func_build_graph.py`
   - builds nodes/edges (`near` relationships using centroid distance threshold)
   - writes `scene_graph.json`
   - creates an LLM prompt: `llm_prompt.txt`
3. `func_query_local_llm.py`
   - runs local HF model (default `Qwen/Qwen2.5-7B-Instruct`)
   - writes structured response: `LLM/llm_response_real.json`
4. `func_visualize_scene_graph.py`
   - visualizes objects, boxes, relations, and labels

### Stage flow (temporal)

1. `func_segment_clustering_temporal.py`
   - processes all `PC_*.pcd` in dataset folder
   - writes per-frame object JSONs in `frames/`
2. `func_build_temporal_graph.py`
   - matches objects across frames with Hungarian assignment
   - computes displacement, velocity class, direction
   - writes `temporal_scene_graph.json`
   - creates temporal LLM prompt: `llm_temporal_prompt.txt`
3. `func_query_llm_temporal.py`
   - runs local HF model for temporal explanation/safety
   - writes:
     - `LLM/llm_temporal_response.json`
     - `LLM/llm_raw_output.txt`
     - `LLM/llm_analysis_report.txt`
4. `func_visualize_temporal.py`
   - visualizes trajectories and tracked movement context

### How LLM is used

- Inference backend: HuggingFace Transformers (`AutoTokenizer`, `AutoModelForCausalLM`).
- Default model in both main scripts: `Qwen/Qwen2.5-7B-Instruct`.
- Model files are cached locally under:
  - `LLM/Model_Cache/`
- Prompts are generated from scene-graph artifacts (not raw point clouds directly).
- LLM is instructed to return strict JSON (for downstream parsing).

### Key configurable parameters

- Single-frame config (`[Main] run_pipeline.py`):
  - `VOXEL_SIZE`, `RANSAC_DISTANCE`, `CLUSTER_EPS`, `CLUSTER_MIN_POINTS`
  - `GRAPH_DIST_THRESHOLD`
  - `LLM_MODEL`
- Temporal config (`[Main] run_temporal_pipeline.py`):
  - same segmentation parameters
  - `MAX_TRACK_DISTANCE`, `MOVEMENT_THRESHOLD`
  - `GRAPH_DIST_THRESHOLD`, `LLM_MODEL`

### Output structure (scene-graph side)

- Single-frame run:
  - `Results/<pcd_name>/scene_objects_clustered.json`
  - `Results/<pcd_name>/scene_graph.json`
  - `Results/<pcd_name>/llm_prompt.txt`
  - `Results/<pcd_name>/LLM/llm_response_real.json`
- Temporal run:
  - `Results/<dataset_name>/frames/frame_XX_objects.json`
  - `Results/<dataset_name>/frame_index.json`
  - `Results/<dataset_name>/temporal_scene_graph.json`
  - `Results/<dataset_name>/llm_temporal_prompt.txt`
  - `Results/<dataset_name>/LLM/llm_temporal_response.json`

## B) MinkUNET Segmentation Pipeline

Detailed docs are in `MinkUNET/README.md`.  
This is the practical run order:

### 1) Build manifests (one-time, re-run if dataset split changes)

```bash
python MinkUNET/scripts/build_manifests.py --config MinkUNET/configs/base.yaml
```

### 2) Convert manual annotations from CloudCompare (re-run when labels change)

```bash
python MinkUNET/scripts/import_cloudcompare_labels.py --config MinkUNET/configs/base.yaml
python MinkUNET/scripts/build_manifests.py --config MinkUNET/configs/base.yaml --labeled-val-ratio 0.2
```

### 3) Contrastive pretraining

```bash
python MinkUNET/scripts/train_contrastive.py --config MinkUNET/configs/pretrain.yaml --skip-val --num-workers 0
```

### 4) Segmentation fine-tuning

```bash
python MinkUNET/scripts/train_segmentation.py --config MinkUNET/configs/finetune.yaml
```

### 5) Validation

```bash
python MinkUNET/scripts/validate_segmentation.py --config MinkUNET/configs/finetune.yaml
```

### 6) Inference (per-frame output folders)

```bash
python MinkUNET/scripts/infer_segmentation.py --config MinkUNET/configs/inference.yaml --checkpoint MinkUNET/checkpoints/semantic_best.pt --output-dir Results/MainRun_Inference
```

Each frame is saved as:

- `Results/MainRun_Inference/<frame_name>/prediction.npz`
- `Results/MainRun_Inference/<frame_name>/prediction.pcd`
- `Results/MainRun_Inference/<frame_name>/summary.json`

## Main Segmentation Classes

- `wall`
- `equipment`
- `human`
- `conveyor`
- `roof`
- `other`

## Training Logs and Plots

Main training runs auto-save timestamped metrics/plots:

- Contrastive:
  - `MinkUNET/logs/contrastive/<timestamp>/...`
- Segmentation:
  - `MinkUNET/logs/segmentation/<timestamp>/...`

Contrastive diagnostics include:

- Similarity histograms (`positive` vs `negative` cosine similarity)
- Layer sparsity/activation summaries per backbone stage

## Paper-Scale Experiments (Separate Pipeline)

If needed, run full ablations and comparisons in isolated folders:

```bash
python MinkUNET/scripts/run_paper_study.py --config MinkUNET/configs/paper_study.yaml
```

Outputs go under:

- `Results/Paper_Study/<run_name_timestamp>/`

This does not interfere with your normal main-run workflow.
