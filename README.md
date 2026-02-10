# 🏭 3D Scene Graph Mining Safety System

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-0.17+-green?logo=3d-rotation&logoColor=white)](http://www.open3d.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Local_LLM-orange)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Automated 3D Scene Understanding + LLM Safety Analysis for Underground Mining**

This project processes raw 3D point cloud data from underground longwall mining environments, constructs a **3D Scene Graph**, and uses a **Local LLM** to identify mining equipment (Shearer, Hydraulic Chocks) and perform safety assessments.

---

## 🎯 Features

- **Geometric Segmentation**: DBSCAN clustering to identify objects from raw point clouds.
- **Scene Graph Construction**: Builds a graph of objects with spatial relationships.
- **Local LLM Reasoning**: Flexible LLM integration (supports DeepSeek, Qwen, Llama, Mistral, etc.).
- **3D Visualization**: Interactive Open3D viewer with colored bounding boxes and relationship arrows.
- **Automated Pipeline**: Single script (`run_pipeline.py`) runs the entire workflow end-to-end.

---

## 📁 Project Structure

```
Scene-Graph-Mining-Safety/
├── [Main] run_pipeline.py            # 🚀 Single-frame pipeline
├── [Main] run_temporal_pipeline.py   # 🚀 Temporal (multi-frame) pipeline
├── func_segment_clustering.py        # Single-frame segmentation
├── func_segment_clustering_temporal.py # Multi-frame segmentation
├── func_build_graph.py               # Single-frame graph construction
├── func_build_temporal_graph.py      # Temporal graph + object tracking
├── func_query_local_llm.py           # LLM inference (single-frame)
├── func_query_llm_temporal.py        # LLM inference (temporal)
├── func_visualize_scene_graph.py     # Single-frame visualization
├── func_visualize_temporal.py        # Temporal visualization
├── LLM/                              # LLM model cache (auto-downloaded)
│   └── Model_Cache/
├── Results/                          # Output folder
│   └── <Dataset_Name>/               # Named by dataset (e.g., "01")
│       ├── frames/                   # Per-frame results (temporal only)
│       ├── temporal_scene_graph.json
│       └── LLM/
├── Datasets/                         # Your point cloud data
│   └── 01/                           # Example multi-frame dataset
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/maninka123/Scene-Graph_Mine-Safety.git
cd Scene-Graph_Mine-Safety
```

### 2. Install Dependencies

```bash
pip install open3d numpy matplotlib transformers torch accelerate
```

### 3. Add Your Point Cloud

Place your `.pcd` file in the `Datasets/` folder and update the path in `[Main] run_pipeline.py`:

```python
class Config:
    PCD_FILE = r"Datasets/pcd/your_file.pcd"
```

### 4. Run the Pipeline

```bash
python "[Main] run_pipeline.py"
```

---

## 🤖 Choosing an LLM Model

The system supports any HuggingFace-compatible LLM. Choose based on your **GPU VRAM**:

| VRAM  | Recommended Model             | HuggingFace ID                              | Size  |
| ----- | ----------------------------- | ------------------------------------------- | ----- |
| 8GB   | DeepSeek-R1-Distill-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | ~3GB  |
| 12GB  | Mistral-7B-Instruct           | `mistralai/Mistral-7B-Instruct-v0.3`        | ~14GB |
| 16GB+ | Qwen2.5-7B-Instruct           | `Qwen/Qwen2.5-7B-Instruct`                  | ~14GB |
| 24GB+ | Llama-3.1-8B-Instruct         | `meta-llama/Llama-3.1-8B-Instruct`          | ~16GB |

### How to Change the Model

1. **Open** `[Main] run_pipeline.py`
2. **Find** the `Config` class (around line 27)
3. **Change** `LLM_MODEL` to your preferred model:

```python
class Config:
    # ...
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Change this line
```

### Parameters to Adjust for Different Models

In `func_query_local_llm.py`, you may need to adjust these parameters based on your model choice:

| Parameter        | Reasoning Models (1.5B) | Instruct Models (7B+) | Why?                                                                                                           |
| ---------------- | ----------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------- |
| `torch_dtype`    | `torch.float16`         | `torch.bfloat16`      | bfloat16 has better numerical stability for larger models                                                      |
| `max_new_tokens` | 2048                    | 1024                  | Reasoning models "think out loud" before answering, needing more tokens. Instruct models output JSON directly. |
| `do_sample`      | `False`                 | `False`               | Deterministic output for consistent JSON                                                                       |

> **💡 Insight**: Smaller "reasoning" models (like DeepSeek-R1) show their chain-of-thought before the final answer, requiring more output tokens. Larger "instruct" models (like Qwen2.5, Llama) follow instructions directly and produce clean JSON without verbose reasoning.

---

## ⚙️ Configuration

All parameters are centralized in `[Main] run_pipeline.py`:

| Parameter              | Default          | Description                          |
| ---------------------- | ---------------- | ------------------------------------ |
| `VOXEL_SIZE`           | 0.05             | Downsampling resolution (meters)     |
| `RANSAC_DISTANCE`      | 0.1              | Floor removal threshold              |
| `CLUSTER_EPS`          | 0.5              | DBSCAN epsilon (cluster distance)    |
| `CLUSTER_MIN_POINTS`   | 50               | Minimum points per cluster           |
| `GRAPH_DIST_THRESHOLD` | 2.5              | Max distance for "near" relationship |
| `LLM_MODEL`            | _(configurable)_ | Local LLM model (see above)          |

---

## 📦 Dependencies

- Python 3.8+
- [Open3D](http://www.open3d.org/) - 3D data processing
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Color mapping
- [Transformers](https://huggingface.co/docs/transformers) - LLM loading
- [PyTorch](https://pytorch.org/) - Deep learning backend
- [Accelerate](https://huggingface.co/docs/accelerate) - Efficient model loading

---

## 🕐 Temporal Pipeline (Multi-Frame Analysis)

The temporal pipeline processes **sequential point cloud frames** to track objects over time and analyze movements.

### Features

- **Multi-Frame Processing**: Processes all PCD files in a dataset folder
- **Object Tracking**: Matches objects across frames using centroid-based Hungarian algorithm
- **Movement Analysis**: Classifies objects as `stationary`, `moving_slow`, or `moving_fast`
- **Temporal LLM Reasoning**: LLM analyzes movement patterns and provides temporal safety insights

### Quick Start

```bash
python "[Main] run_temporal_pipeline.py"
```

### Configuration

Edit `[Main] run_temporal_pipeline.py`:

```python
class TemporalConfig:
    DATASET_FOLDER = r"Datasets\01"   # Folder with PCD sequence
    DATASET_NAME = "01"               # Output folder name
    MAX_TRACK_DISTANCE = 1.0          # Max distance for object matching
    MOVEMENT_THRESHOLD = 0.2          # Min displacement to count as "moving"
```

### Output Structure

```
Results/01/
├── frames/
│   ├── frame_00_objects.json
│   ├── frame_00_clusters.pcd
│   └── ...
├── frame_index.json
├── temporal_scene_graph.json
├── llm_temporal_prompt.txt
├── LLM/
│   └── llm_temporal_response.json
└── temporal_visualization.pcd
```

### Temporal Files

| File                                  | Description                            |
| ------------------------------------- | -------------------------------------- |
| `[Main] run_temporal_pipeline.py`     | Main entry point for temporal analysis |
| `func_segment_clustering_temporal.py` | Multi-frame segmentation               |
| `func_build_temporal_graph.py`        | Object tracking & movement analysis    |
| `func_query_llm_temporal.py`          | Temporal LLM inference                 |
| `func_visualize_temporal.py`          | Movement trajectory visualization      |

---

## 🔮 Future Work

- [ ] Real-time streaming point cloud processing
- [ ] Web-based visualization dashboard
- [x] ~~Multi-frame temporal scene graphs~~ ✅ Implemented!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Open3D](http://www.open3d.org/) for 3D processing
- [HuggingFace](https://huggingface.co/) for LLM infrastructure
- Mining safety research community

---

**Developed for safer underground mining operations.**
