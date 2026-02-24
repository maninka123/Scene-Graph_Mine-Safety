# MinkUNET Semantic Segmentation Pipeline

This folder provides a full segmentation pipeline for colored 3D point clouds with:

- Contrastive pretraining on unlabeled `.pcd` frames.
- Cluster-assisted annotation for a small sample subset.
- Supervised fine-tuning for semantic segmentation.
- Validation with IoU metrics.
- Inference on single frames or sequences with optional temporal smoothing.

## Folder Layout

```text
MinkUNET/
  configs/
  scripts/
  minkunet/
  checkpoints/
  logs/
  data/
```

## Expected Data Paths

By default this pipeline reads from:

- Unlabeled frames: `Datasets/Data_all/*.pcd`
- Annotation workspace: `Datasets/Annotated/`

It also checks `Dataset/Data_all` as a fallback if present.

## Install

```bash
pip install -r MinkUNET/requirements.txt
```

If you want the real sparse MinkUNet model, install MinkowskiEngine for your CUDA/PyTorch setup.

## 1) Create Annotation Sample

Copies a small subset of frames into `Datasets/Annotated/sample`.

```bash
python MinkUNET/scripts/create_annotation_sample.py --count 20
```

## 2) Create Cluster Proposals for Manual Labeling

Runs DBSCAN clustering for each sample frame and creates:

- `*_cluster_proposal.npz` (points/colors/cluster_ids)
- `*_cluster_labels.json` (fill in `manual_label` per cluster)
- `*_clusters_preview.pcd` (colored preview)

```bash
python MinkUNET/scripts/cluster_for_annotation.py
```

## 3) Fill Manual Labels

Two supported annotation workflows:

1. Cluster template workflow:
Edit `Datasets/Annotated/cluster_proposals/*_cluster_labels.json` and set `manual_label`.

2. CloudCompare workflow:
Create per-frame files in `Datasets/Annotated/manual_cc/<frame_name>/`:
`wall.pcd`, `equipment.pcd`, `human.pcd` (optional), `conveyor.pcd`, `roof.pcd`, `other.pcd`.

Current class taxonomy:
`wall`, `equipment`, `human`, `conveyor`, `roof`, `other`

## 4) Materialize Point Labels

From cluster templates:

```bash
python MinkUNET/scripts/apply_cluster_labels.py
```

From CloudCompare per-class PCDs:

```bash
python MinkUNET/scripts/import_cloudcompare_labels.py
```

## 5) Build Manifests

Creates train/val manifests for unlabeled and labeled sets.

```bash
python MinkUNET/scripts/build_manifests.py
```

## 6) Contrastive Pretraining

```bash
python MinkUNET/scripts/train_contrastive.py --config MinkUNET/configs/pretrain.yaml
```

## 7) Segmentation Fine-Tuning

```bash
python MinkUNET/scripts/train_segmentation.py --config MinkUNET/configs/finetune.yaml
```

## 8) Validation

```bash
python MinkUNET/scripts/validate_segmentation.py --config MinkUNET/configs/finetune.yaml --checkpoint MinkUNET/checkpoints/semantic_best.pt
```

Validation artifacts now include:

- `metrics.json`
- `per_class_iou.png`
- `confusion_matrix.npy`
- `confusion_matrix.json`
- `confusion_matrix_counts.png`
- `confusion_matrix_normalized.png`

## 9) Inference

Single frame:

```bash
python MinkUNET/scripts/infer_segmentation.py --checkpoint MinkUNET/checkpoints/semantic_best.pt --input Datasets/Data_all/PC_20260122_153641662.pcd
```

Sequence with temporal smoothing:

```bash
python MinkUNET/scripts/infer_segmentation.py --checkpoint MinkUNET/checkpoints/semantic_best.pt --input Datasets/Data_all --temporal-window 5
```

Inference outputs are now grouped per frame:

- `Results/MinkUNET_inference/<frame_name>/prediction.npz`
- `Results/MinkUNET_inference/<frame_name>/prediction.pcd`
- `Results/MinkUNET_inference/<frame_name>/summary.json`

## 10) Single Cloud Preview (Interactive)

Run one-frame inference and open an Open3D window for visual inspection:

```bash
python MinkUNET/scripts/preview_single_prediction.py --input Datasets/Data_all/PC_20260122_153641662.pcd
```

## One Command Orchestrator

```bash
python MinkUNET/scripts/run_full_pipeline.py
```

This orchestrator runs setup stages and prints where manual annotation is required.

## Paper Study Pipeline (Separate, Non-Interfering)

Run all paper experiments (scratch vs pretrained, ablations, runtime, robustness)
in isolated folders under `Results/Paper_Study/`:

```bash
python MinkUNET/scripts/run_paper_study.py --config MinkUNET/configs/paper_study.yaml
```

Generate/refresh comparison plots from a study folder:

```bash
python MinkUNET/scripts/visualize_paper_results.py --study-dir Results/Paper_Study/<your_run_folder>
```

## Training Outputs For Papers

Each training run now creates a timestamped folder under:

- Contrastive: `MinkUNET/logs/contrastive/<run_timestamp>/`
- Segmentation: `MinkUNET/logs/segmentation/<run_timestamp>/`

Inside each run folder:

- `metrics/epoch_metrics.jsonl`
- `metrics/epoch_metrics.csv`
- `plots/*.png`
- `artifacts/resolved_config.json`
- `artifacts/run_summary.json`

Contrastive runs also include diagnostics:

- `diagnostics/similarity_metrics.jsonl`
- `diagnostics/similarity_metrics.csv`
- `diagnostics/layer_sparsity_metrics.jsonl`
- `diagnostics/layer_sparsity_metrics.csv`
- `diagnostics/similarity_hist_epoch_XXX.png`
- `diagnostics/layer_sparsity_epoch_XXX.png`

Typical plots generated:

- Contrastive: loss curve, learning-rate curve
- Segmentation: loss, mIoU, accuracy, LR, and per-class IoU curves
