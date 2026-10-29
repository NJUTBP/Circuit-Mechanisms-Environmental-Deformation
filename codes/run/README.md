# Model simulation

`run_pipeline.py` runs the model through initialization, baseline, initial
exposure, and familiarization. Experiment settings are in `configs/`.

From this directory:

```bash
python run_pipeline.py --help
python run_pipeline.py plan
python run_pipeline.py run --stage init --task-index 0
```

`--task-index` selects a task from the stage plan. Use `--config` to select a
configuration, for example:

```bash
python run_pipeline.py --config configs/config.model.json plan
```

The default configuration uses 8 independent networks, each with three 64 × 64
grid modules. Initial activity is 1 at `(32, 15)`, `(32, 32)`, and `(32, 49)`,
with uniform background activity in `[0, 0.01)`. Warm-up velocities are sampled
in `(-0.01, 0.01)`, and velocity rotations are `[-15, 15, -10]` degrees.

Set `RESCALE_DATA_ROOT` and `RESCALE_OUTPUT_ROOT` to choose data and output
locations. Seeds, learning rates, network sizes, and session settings are
specified in the JSON configurations.

`prepare_sensitivity_experiments.py`, `prepare_learning_rate_control.py`, and
`prepare_place_cell_lr_zero.py` prepare parameter-sweep configurations.
`run_sensitivity_task.py` executes a selected task. Use each program's `--help`
option for arguments.
