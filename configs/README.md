# Experiment Configs

These files are shell environment configs for the existing PACE-style runners. They are intentionally not YAML configs: the repository entrypoints already read environment variables.

Use them like this:

```bash
set -a
source configs/thesis_runs/main_primary.env
set +a
bash run_main_experiments.sh
```

The `thesis_runs/` configs cover the main reported experiment families. The `sweeps/` configs document ablations used in the thesis, such as consistency-weight and IVON posterior-strength sweeps.

