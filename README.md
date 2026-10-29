# Circuit mechanisms shaping place- and grid-cell representations under environmental deformation

Code and plotting data for the manuscript submitted to *Communications in Theoretical Physics*.

[中文说明](README_ZH.md) · [Figure data index](docs/DATA_INDEX.md) · [MIT license](LICENSE)

## Contents

| Directory | Contents |
| --- | --- |
| `codes/run/` | Model simulation code and experiment configurations |
| `codes/analysis/` | Network analysis, statistics, and transfer entropy |
| `Figs_CTP/Plot_code/` | Plotting scripts |
| `Figs_CTP/Plot_data/` | Source data organized by figure number |
| `Figs_CTP/Main_figs/`, `Figs_CTP/Supp_figs/` | 6 main figures and 16 supplementary figures |

## Plot the figures

Use Python 3.11. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-plot.txt
python Figs_CTP/Plot_code/plot_all.py
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.
For Main Figure 1, install Poppler and add `pdftoppm` to PATH, or set `PDFTOPPM`
to the executable path. Use Times New Roman for the figure font.

Figures are saved to `Figs_CTP/Reproduced/`. To plot a single figure:

```bash
python Figs_CTP/Plot_code/plot_all.py --figures main_figure_2
```

See [DATA_INDEX.md](docs/DATA_INDEX.md) for the figure-to-file mapping and
[DATA_FILES.csv](docs/DATA_FILES.csv) for table fields and array dimensions.

## Simulation and analysis

In a separate Python environment:

```bash
python -m pip install -r requirements-analysis.txt
python codes/run/run_pipeline.py --help
```

See [model instructions](codes/run/README.md) and
[analysis instructions](codes/analysis/README.md) for entry points and configuration.
The transfer-entropy package can be installed with:

```bash
python -m pip install -e codes/analysis/te_population_gcmi
te-gcmi --help
```

## License

[MIT](LICENSE).
