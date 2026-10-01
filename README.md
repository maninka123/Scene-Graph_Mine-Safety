# [From 3D Perception to Safety Reasoning](https://www.sciencedirect.com/science/article/pii/S0957417426035050)

> Research code for 3D perception, scene graphs, and safety reasoning in underground mines.

[![Published in](https://img.shields.io/badge/Published%20in-Expert%20Systems%20with%20Applications-0072b1.svg)](https://www.sciencedirect.com/journal/expert-systems-with-applications)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![Qwen](https://img.shields.io/badge/LLM-Qwen2.5--3B-6f42c1.svg)](configs/paper.yaml)
[![MinkowskiEngine](https://img.shields.io/badge/3D%20perception-MinkowskiEngine-2b6cb0.svg)](src/mine_safety/perception/network.py)
[![ROS-free demo](https://img.shields.io/badge/demo-ROS--free-2c7a7b.svg)](#minegraph-studio)

![Graphical abstract of the perception-to-reasoning pipeline](assets/figures/Graphical%20Abstract_final.png)

This repository contains the software pipeline for **[“From 3D Perception to Safety Reasoning: An
Intelligent Hybrid Framework for Real-Time Underground Mine Monitoring”](https://www.sciencedirect.com/science/article/pii/S0957417426035050)**, published in *Expert Systems
with Applications* (2025 Impact Factor: 9.4; CiteScore: 17.0; Scopus Q1, 97th percentile).
The repository provides the architecture, training and inference entry points, scene and temporal
graph construction, deterministic safety rules, grounded local-LLM reasoning, GraphRAG memory, tests, and
an interactive visualiser. Experimental results and ablation studies are not included in this repository.

## Scope and safety

This is research software, not a certified or operational safety system. The paper pipeline is implemented
in Python, including temporal tracking. The browser app is deliberately narrower: it visualises one frame
and one scene graph so users can inspect the processing stages. The app does **not** create a temporal
graph and is not connected to a simulator, live sensors, alarms, machinery, or operational controls.

## Paper pipeline

| Stage | Implementation |
| --- | --- |
| Sparse 3D perception | XYZRGB sparse input, MinkUNet encoder–decoder, shared 96-D representation, contrastive and semantic heads |
| Uncertainty | Predictive entropy, morphological closing, removal of connected components smaller than 20 voxels, DBSCAN proposals, and paper merge criteria |
| Scene graph | Object/anomaly nodes and directed proximity relations within 8 m |
| Temporal graph | Hungarian association, 10 s rolling history, velocity, and movement state |
| Safety rules | Proximity, TTC, blind spot, congestion, and low visibility |
| Context reasoning | Appendix A prompt contract with schema and object-ID validation |
| Longitudinal memory | Selective GraphRAG retrieval through LlamaIndex and local Qdrant with Qwen3 embedding and reranking |

All traceable constants are collected in [`configs/paper.yaml`](configs/paper.yaml). The executable modules
live under [`src/mine_safety`](src/mine_safety), and the training/inference entry points are in
[`scripts`](scripts).

## MineGraph Studio

MineGraph Studio is a React, Three.js, and FastAPI demonstrator for inspecting the pipeline with a single
scene graph. It includes an interactive point cloud, detection boxes, graph controls, deterministic-rule
evidence, execution timing, optional local Qwen reasoning, and local point-cloud upload support.

![MineGraph Studio single-scene pipeline demonstrator](assets/figures/minegraph-studio.png)

### Start on Windows

Double-click **`Start MineGraph Studio.cmd`** in the repository folder. On the first launch it installs any
missing app packages and builds the web interface; later launches reuse that build. The browser opens when
the local API is ready. Keep the launcher window open while using the app and press `Ctrl+C` to stop it.

### Manual start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[app,llm,dev]"
cd web
npm install
npm run build
cd ..
python scripts/build_demo_bundle.py
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Uploaded `.pcd`, `.ply`, and `.npz` files remain local. GPU segmentation
requires CUDA, MinkowskiEngine, and a compatible local checkpoint. Qwen weights are never downloaded
silently; set `MINEGRAPH_QWEN_MODEL` when using a custom local model path.

## Run the Python pipeline

```bash
pip install -e .
mine-safety examples/scene_graph.json --output outputs/assessment.json
```

Enable local contextual reasoning after installing the LLM dependencies:

```bash
pip install -e ".[llm]"
mine-safety examples/scene_graph.json --llm
```

To use persistent historical memory with Qwen3 embeddings, LlamaIndex, Qdrant, and Qwen3 reranking:

```bash
mine-safety examples/scene_graph.json --llm --qdrant-path ./memory/qdrant
```

Reuse the same Qdrant path across runs to retrieve earlier graph memories. Without `--qdrant-path`,
the CLI uses the lightweight in-memory archive for demonstrations and tests.

For point-cloud training and inference, install PyTorch and a compatible MinkowskiEngine build first:

```bash
pip install -e ".[pointcloud]"
python scripts/train_contrastive.py data/unlabelled.jsonl
python scripts/train_semantic.py data/train.jsonl --val-manifest data/val.jsonl
python scripts/infer_point_cloud.py data/example.pcd \
  --checkpoint checkpoints/semantic_nine_class.pt
```

Training `.npz` files contain `points [N,3]`, `colors [N,3]` in `[0,1]`, and supervised files also contain
integer `labels [N]`. JSONL manifests use `{"path":"relative/or/absolute/scene.npz"}`.
Keep training and validation point clouds in separate manifests; checkpoint selection and early stopping
use validation loss.

## Paper figures

The following are author-supplied paper assets, not results regenerated by this repository.

<table>
  <tr>
    <td align="center" bgcolor="#ffffff"><img src="assets/figures/NN%20model_v2.png" alt="Minkowski UNet architecture"><br><sub>Sparse 3D network</sub></td>
    <td align="center" bgcolor="#ffffff"><img src="assets/figures/anaomaly%20figure%20v2.png" alt="Entropy anomaly proposal pipeline"><br><sub>Uncertainty and anomaly proposals</sub></td>
  </tr>
  <tr>
    <td align="center" bgcolor="#ffffff"><img src="assets/figures/LLM%20Pipline%20v2.png" alt="GraphRAG reasoning pipeline"><br><sub>Selective longitudinal reasoning</sub></td>
    <td align="center" bgcolor="#ffffff"><img src="assets/figures/hazards%20in%20tunnel.png" alt="Graph-grounded mine hazard examples"><br><sub>Graph-grounded hazard examples</sub></td>
  </tr>
</table>

> The supplied anomaly panel shows an earlier voxel setting. The executable configuration follows the
> paper text and uses 0.01 m voxels. The original figure is retained unchanged.

## Validation and deployment

```bash
pip install -e ".[dev]"
pytest -q
```

GitHub Pages can host the read-only frontend bundle. Live Python graph processing, CUDA perception, local
Qwen inference, and Qdrant require a suitable backend and cannot run directly in GitHub Pages.

## Cite

Published in *[Expert Systems with Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications)*
(October 2026), article 134601. [Read the paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417426035050)
· [DOI: 10.1016/j.eswa.2026.134601](https://doi.org/10.1016/j.eswa.2026.134601).

```bibtex
@article{ranasinghe2026from3d,
  title   = {From 3D Perception to Safety Reasoning: An Intelligent Hybrid Framework for Real-Time Underground Mine Monitoring},
  author  = {Ranasinghe, Pasindu and Raval, Simit and Patra, Dibyayan and Banerjee, Bikram and Canbulat, Ismet},
  journal = {Expert Systems with Applications},
  year    = {2026},
  eid     = {134601},
  doi     = {10.1016/j.eswa.2026.134601},
  url     = {https://www.sciencedirect.com/science/article/pii/S0957417426035050}
}
```

Machine-readable metadata is available in [`CITATION.cff`](CITATION.cff).
