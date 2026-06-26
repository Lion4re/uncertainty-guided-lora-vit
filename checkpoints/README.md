# Checkpoints

Model checkpoints are not stored in this repository because they are large generated artifacts.

Regenerate checkpoints with the commands in `docs/reproduce_thesis_results.md`, or place existing
checkpoints under local `outputs/` or `outs/` directories. Those directories are ignored by Git.

The pretrained ViT-B/16 checkpoint expected by the training and evaluation scripts is:

```text
ViT-B_16.npz
```

Place it at the repository root. It is also ignored by Git.
