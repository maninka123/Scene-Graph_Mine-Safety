# 🏭 3D Scene Graph Mining Safety System

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-0.17+-green?logo=3d-rotation&logoColor=white)](http://www.open3d.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--R1-orange)](https://huggingface.co/deepseek-ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Automated 3D Scene Understanding + LLM Safety Analysis for Underground Mining**

This project processes raw 3D point cloud data from underground longwall mining environments, constructs a **3D Scene Graph**, and uses a **Local LLM (DeepSeek-R1)** to identify mining equipment (Shearer, Hydraulic Chocks) and perform safety assessments.

---

## 🎯 Features

- **Geometric Segmentation**: DBSCAN clustering to identify objects from raw point clouds.
- **Scene Graph Construction**: Builds a graph of objects with spatial relationships.
- **Local LLM Reasoning**: Uses DeepSeek-R1-Distill-Qwen-1.5B for object identification and safety analysis.
- **3D Visualization**: Interactive Open3D viewer with colored bounding boxes and relationship arrows.
- **Automated Pipeline**: Single script (`run_pipeline.py`) runs the entire workflow end-to-end.

---

## 📁 Project Structure

```
Scene-Graph-Mining-Safety/
├── [Main] run_pipeline.py      # 🚀 Main entry point - run this!
├── func_segment_clustering.py  # Geometric segmentation (DBSCAN)
├── func_build_graph.py         # Scene graph construction
├── func_query_local_llm.py     # Local LLM inference (DeepSeek)
├── func_visualize_scene_graph.py # 3D visualization
├── LLM/                        # LLM model cache (auto-downloaded)
│   └── DeepSeek_Model/         # Model weights (gitignored)
├── Results/                    # Output folder (per point cloud)
│   └── <PCD_Name>/
│       ├── scene_objects_clustered.json
│       ├── scene_graph.json
│       ├── llm_prompt.txt
│       ├── colored_clusters.pcd    # Colored cluster output
│       └── LLM/
│           └── llm_response_real.json
├── Datasets/                   # Your point cloud data (gitignored)
│   └── pcd/
│       └── your_file.pcd
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Scene-Graph-Mining-Safety.git
cd Scene-Graph-Mining-Safety
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

The pipeline will:

1. 📊 Show the **Original Point Cloud**
2. 🎨 Show **Clustered Objects** (colored)
3. 🤖 Run **LLM Analysis** (first run downloads model ~3GB)
4. 🗺️ Show **Final Scene Graph** with annotations

---

## 🖼️ Sample Output

| Step               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| **1. Raw PCD**     | Original point cloud before processing                       |
| **2. Clusters**    | DBSCAN segmentation with colored objects                     |
| **3. Scene Graph** | Bounding boxes (🔴 Shearer, 🟢 Chocks) + relationship arrows |

---

## ⚙️ Configuration

All parameters are centralized in `[Main] run_pipeline.py`:

| Parameter              | Default                                     | Description                          |
| ---------------------- | ------------------------------------------- | ------------------------------------ |
| `VOXEL_SIZE`           | 0.05                                        | Downsampling resolution (meters)     |
| `RANSAC_DISTANCE`      | 0.1                                         | Floor removal threshold              |
| `CLUSTER_EPS`          | 0.5                                         | DBSCAN epsilon (cluster distance)    |
| `CLUSTER_MIN_POINTS`   | 50                                          | Minimum points per cluster           |
| `GRAPH_DIST_THRESHOLD` | 2.5                                         | Max distance for "near" relationship |
| `LLM_MODEL`            | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Local LLM model                      |

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

## 🔮 Future Work

- [ ] Support for more LLM models (Llama, Mistral)
- [ ] Real-time streaming point cloud processing
- [ ] Web-based visualization dashboard
- [ ] Multi-frame temporal scene graphs

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Open3D](http://www.open3d.org/) for 3D processing
- [DeepSeek](https://www.deepseek.com/) for the local LLM
- Mining safety research community

---

**Made with ❤️ for safer underground mining operations.**
