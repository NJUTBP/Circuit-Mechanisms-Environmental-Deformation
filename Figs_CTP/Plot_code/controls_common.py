"""Shared typography, axis-fitting and export functions."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def configure() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "mathtext.cal": "Times New Roman",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def five_ticks(low: float, high: float, reference: float | None = None) -> np.ndarray:
    """Five readable ticks enclosing all network values and plotted intervals."""
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError(f"Invalid y range: {low}, {high}")
    span = max(high - low, 1e-4)
    # Choose the tick grid from the data range; add breathing space to the
    # axis limits later so a value near a round tick cannot double the span.
    bottom, top = (low, high) if high > low else (low - 0.02 * span, high + 0.02 * span)
    needed = (top - bottom) / 4
    exponent = int(np.floor(np.log10(needed)))
    steps = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]
    candidates = []
    for power in range(exponent - 1, exponent + 3):
        for mantissa in steps:
            step = mantissa * 10 ** power
            if 4 * step < top - bottom - 1e-10:
                continue
            first = int(np.floor(bottom / step))
            for index in range(first - 4, first + 2):
                ticks = (index + np.arange(5, dtype=float)) * step
                if ticks[0] > bottom + 1e-10 or ticks[-1] < top - 1e-10:
                    continue
                missing_reference = (
                    reference is not None and bottom <= reference <= top
                    and not np.isclose(ticks, reference, atol=1e-9).any()
                )
                excess = (ticks[-1] - ticks[0]) - (top - bottom)
                center = abs((ticks[0] + ticks[-1]) / 2 - (bottom + top) / 2)
                candidates.append((int(missing_reference), excess, center, step, ticks))
    if not candidates:
        raise RuntimeError("Could not place five y ticks")
    # Equivalent decimal grids can differ by a few binary floating-point bits.
    # Resolve those ties by centering the data rather than choosing an edge.
    ticks = min(candidates, key=lambda item: (
        item[0], round(item[1], 9), round(item[2], 9), item[3]))[-1]
    return np.round(ticks, 10)


def apply_five_ticks(ax: plt.Axes, low: float, high: float,
                     reference: float | None = None) -> None:
    ticks = five_ticks(low, high, reference)
    margin = 0.02 * max(high - low, 1e-4)
    ax.set_ylim(min(float(ticks[0]), low - margin),
                max(float(ticks[-1]), high + margin))
    ax.set_yticks(ticks)
    ax.tick_params(direction="out", width=0.6, length=2.5, pad=2)


def tight_axis(ax):
    """Enclose every drawn network/mean/CI; never force a distant reference."""
    visible = []
    for line in ax.lines:
        if line.get_zorder() > 0:
            values = np.asarray(line.get_ydata(), float)
            visible.extend(values[np.isfinite(values)].tolist())
    for collection in ax.collections:
        for path in collection.get_paths():
            values = path.vertices[:, 1]
            visible.extend(values[np.isfinite(values)].tolist())
    if not visible:
        raise ValueError("No observed network values or confidence interval in panel")
    low, high = float(min(visible)), float(max(visible))
    span = max(high - low, abs((low + high) / 2) * .005, 1e-4)
    kept_references, hidden_references = [], []
    for line in ax.lines:
        if line.get_zorder() != 0:
            continue
        x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        if len(y) == 2 and np.isclose(y[0], y[1]) and np.allclose(x, [0., 1.]):
            reference = float(y[0])
            if low - .10 * span <= reference <= high + .10 * span:
                low, high = min(low, reference), max(high, reference)
                kept_references.append(reference)
            else:
                line.set_visible(False)
                hidden_references.append(reference)
    margin = .055 * max(high - low, span)
    ylim = (low - margin, high + margin)
    ax.set_ylim(*ylim)
    candidates = []
    fallback = []
    require_one_tick = 1.0 in kept_references and ylim[0] <= 1.0 <= ylim[1]
    for bins in range(4, 10):
        locator = MaxNLocator(nbins=bins, min_n_ticks=2,
                             steps=[1, 2, 2.5, 5, 10])
        full_ticks = locator.tick_values(*ylim)
        ticks = full_ticks
        ticks = ticks[(ticks >= ylim[0] - 1e-12) & (ticks <= ylim[1] + 1e-12)]
        has_reference = not require_one_tick or np.isclose(ticks, 1., atol=1e-12).any()
        if 4 <= len(ticks) <= 6 and has_reference:
            coverage = (ticks[-1] - ticks[0]) / (ylim[1] - ylim[0])
            candidates.append((abs(len(ticks) - 5), -coverage, ticks))
        # A tight interval can fall between standard decimal grids (e.g.
        # 0.91--1.28). Expand by only the nearest round boundary if needed to
        # obtain four readable ticks; avoid nonstandard0.075 steps.
        for lower in (ylim[0], float(full_ticks[0])):
            for upper in (ylim[1], float(full_ticks[-1])):
                if lower > ylim[0] or upper < ylim[1]:
                    continue
                trial = full_ticks[(full_ticks >= lower - 1e-12) & (full_ticks <= upper + 1e-12)]
                has_reference = not require_one_tick or np.isclose(trial, 1., atol=1e-12).any()
                if 4 <= len(trial) <= 6 and has_reference:
                    expansion = (upper - lower) - (ylim[1] - ylim[0])
                    fallback.append((expansion, abs(len(trial) - 5), trial, (lower, upper)))
    if not candidates:
        if not fallback:
            raise RuntimeError(f"Cannot place4--6 standard ticks for{ylim}")
        _, _, ticks, ylim = min(fallback, key=lambda item: item[:2])
    else:
        ticks = min(candidates, key=lambda item: item[:2])[-1]
    ticks[np.isclose(ticks, 0., atol=1e-12)] = 0.
    ax.set_yticks(ticks)
    # Setting FixedLocator can expand limits; restore the deliberately tight view.
    ax.set_ylim(*ylim)
    assert ylim[0] <= min(visible) and ylim[1] >= max(visible)
    return {"visible_data_and_CI_min": float(min(visible)),
            "visible_data_and_CI_max": float(max(visible)),
            "ylim": list(ylim), "yticks": ticks.tolist(),
            "tick_count": len(ticks), "reference_lines_visible": kept_references,
            "reference_lines_hidden_as_distant": hidden_references,
            "all_network_values_means_and_CI_enclosed": True}


def fit_panel(ax):
    """Use every displayed line and interval vertex; never share panel bounds."""
    values = [np.asarray(line.get_ydata(), float).ravel() for line in ax.lines]
    for collection in ax.collections:
        if hasattr(collection, "get_paths"):
            for path in collection.get_paths():
                if len(path.vertices):
                    values.append(np.asarray(path.vertices[:, 1], float))
    all_values = np.concatenate(values)
    assert np.isfinite(all_values).all() and len(all_values)
    low, high = float(all_values.min()), float(all_values.max())
    span = high - low
    if span <= 0:
        span = max(abs(low) * .05, .01)
    lo, hi = low - .06 * span, high + .06 * span
    assert hi > lo and (hi - lo) / 4 > 0
    power = int(np.floor(np.log10((hi - lo) / 4)))
    candidates = []
    for exponent in range(power - 2, power + 3):
        for mantissa in (1., 2., 2.5, 5., 10.):
            step = mantissa * 10. ** exponent
            start = int(np.ceil(lo / step - 1e-10))
            stop = int(np.floor(hi / step + 1e-10))
            count = stop - start + 1
            if 4 <= count <= 6:
                ticks = np.arange(start, stop + 1, dtype=float) * step
                edge_balance = abs((ticks[0] - lo) - (hi - ticks[-1])) / (hi - lo)
                candidates.append((abs(count - 5), edge_balance, -step, ticks))
    if candidates:
        ticks = min(candidates, key=lambda item: item[:3])[-1]
    else:
        # If adjacent conventional step sizes yield only3 or7 interior ticks,
        # expand to the nearest conventional enclosing grid with4–6 ticks.
        for exponent in range(power - 2, power + 3):
            for mantissa in (1., 2., 2.5, 5., 10.):
                step = mantissa * 10. ** exponent
                start, stop = int(np.floor(lo / step)), int(np.ceil(hi / step))
                count = stop - start + 1
                if 4 <= count <= 6:
                    ticks = np.arange(start, stop + 1, dtype=float) * step
                    candidates.append((ticks[-1] - ticks[0], abs(count - 5), step, ticks))
        assert candidates, (low, high)
        ticks = min(candidates, key=lambda item: item[:3])[-1]
        lo, hi = min(lo, float(ticks[0])), max(hi, float(ticks[-1]))
    ticks = np.round(ticks, 10)
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.tick_params(direction="out", width=.6, length=2.5, pad=2)
    assert lo < low <= high < hi and 4 <= len(ticks) <= 6
    return {"displayed_min": low, "displayed_max": high, "ylim": list(ax.get_ylim()),
            "y_ticks": list(map(float, ax.get_yticks())), "tick_count": len(ticks),
            "all_displayed_values_enclosed": True, "padding_fraction": .06}


def save_figure(fig, output_dir: Path, stem: str):
    """Export publication rasters and editable vector figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "tiff", "pdf", "svg"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=600,
                    bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)
    return output_dir / stem
