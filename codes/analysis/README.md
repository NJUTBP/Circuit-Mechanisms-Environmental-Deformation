# Analysis

| Directory | Contents |
| --- | --- |
| `cluster_pipeline/three_experiments/` | Learning-rate, place-cell-count, and grid-spacing analyses |
| `cluster_pipeline/analyze_*.py` | Analysis entry points for the parameter sweeps |
| `local_network_analysis/` | Grid rescaling, phase shifts, field extent, and spatial information |
| `main_run_prev/analysis/` | Rate-map, field-length, and phase-shift functions |
| `te_population_gcmi/` | Population GCMI transfer-entropy analysis |

Use the individual programs' `--help` options to select data paths, output
locations, networks, and analysis parameters. For example:

```bash
python local_network_analysis/initial_exposure_metrics.py --help
python cluster_pipeline/analyze_learning_rates.py --help
```

See the [transfer-entropy package instructions](te_population_gcmi/README.md)
for its installation and command-line interface.
Figure scripts and source tables are in [`../../Figs_CTP/`](../../Figs_CTP/README.md).
