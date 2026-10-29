# Population transfer-entropy analysis

Behavior-adjusted population GCMI transfer-entropy analysis.

From the repository root:

```bash
python -m pip install -e codes/analysis/te_population_gcmi
te-gcmi --help
```

The package requires Python >=3.10, NumPy, and pandas. See
[`config.example.json`](config.example.json) for configuration fields.
The `drivers/` directory contains analysis entry points for specific experiments.
