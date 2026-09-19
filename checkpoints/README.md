# Checkpoints

- `contrastive_backbone.pt` is copied from the original workspace. Its 96-dimensional backbone is
  architecturally compatible with the paper's contrastive stage and can initialise nine-class fine-tuning.
- `legacy_six_class/semantic_best.pt` is preserved for provenance only. It was trained with the old
  six-class taxonomy and is intentionally not loaded by the paper-aligned inference path.
- A valid paper-aligned segmentation checkpoint must contain a nine-output semantic head. Create it with
  `python scripts/train_semantic.py <manifest>`; the default output is `semantic_nine_class.pt`.

No benchmark or claimed paper result is bundled or regenerated here.
