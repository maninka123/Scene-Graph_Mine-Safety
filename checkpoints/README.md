# Checkpoints

- `contrastive_backbone.pt` is copied from the original workspace. Its 96-dimensional backbone is
  architecturally compatible with the paper's contrastive stage and can initialise nine-class fine-tuning.
- `demo_adapter/semantic_best.pt` is a historical app checkpoint preserved for provenance. It is isolated
  from and is not loaded by the paper-aligned inference path.
- A valid paper-aligned segmentation checkpoint must contain a nine-output semantic head. Create it with
  `python scripts/train_semantic.py <manifest>`; the default output is `semantic_nine_class.pt`.

No benchmark or claimed paper result is bundled or regenerated here.
