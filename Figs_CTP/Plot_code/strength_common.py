"""Coupling-strength response heatmaps."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors,font_manager
import numpy as np
import pandas as pd
from archive_common import DATA,export
WIDTH_MM=180.0
W_PG=(1.,2.,3.,4.,5.)
W_GP=(.01,.02,.03,.04,.05)
X_LABEL=r"$W^{\mathrm{GP}}$"
Y_LABEL=r"$W^{\mathrm{PG}}$"
def matrix_for(group, population, module, metric):
    selected = group[(group.population == population) & (group.module == module)
                     & (group.metric == metric)].copy()
    selected["w_pg"] = selected["w_pg"].round(10)
    selected["w_gp"] = selected["w_gp"].round(10)
    pairs = {(round(float(a), 10), round(float(b), 10))
             for a, b in zip(selected.w_pg, selected.w_gp)}
    expected = {(a, b) for a in W_PG for b in W_GP}
    if len(selected) != 25 or pairs != expected:
        raise ValueError(f"Incomplete 5x5 grid: {population}/{module}/{metric}")
    values = selected.pivot(index="w_pg", columns="w_gp", values="mean").reindex(
        index=W_PG, columns=W_GP).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite plotted group estimate: {population}/{module}/{metric}")
    return values, selected

def apply_style():
    # Refuse a silent font substitution. All text, including math letters, uses
    # Times New Roman; editable text is retained in both vector formats.
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.5,
        "axes.labelsize": 10,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.cal": "Times New Roman:italic",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return font_path

def panel(fig, slot, values, title, subtitle, letter):
    inner = slot.subgridspec(1, 2, width_ratios=(1, 0.048), wspace=0.12)
    axis = fig.add_subplot(inner[0])
    color_axis = fig.add_subplot(inner[1])
    low, high = float(values.min()), float(values.max())
    if low == high:
        pad = max(abs(low) * 0.01, 1e-6)
        low, high = low - pad, high + pad
    # Per-panel linear color scaling deliberately follows the original SI,
    # including signed deltas. The full finite range is always displayed.
    norm = colors.Normalize(vmin=low, vmax=high, clip=False)
    image = axis.imshow(values, cmap="viridis", norm=norm, origin="upper",
                        aspect="auto", interpolation="nearest")
    axis.set_xticks(range(5), [f"{v:.2f}" for v in W_GP])
    axis.set_yticks(range(5), [f"{v:g}" for v in W_PG])
    axis.set_ylim(4.5, -0.5)
    axis.set_xlabel(X_LABEL, labelpad=1)
    axis.set_ylabel(Y_LABEL, labelpad=2)
    axis.tick_params(direction="out", length=2.3, width=0.6, pad=2)
    axis.set_xticks(np.arange(0.5, 4.5, 1), minor=True)
    axis.set_yticks(np.arange(0.5, 4.5, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.text(0.5, 1.14, title, transform=axis.transAxes, ha="center", va="bottom",
              fontsize=9.5, fontweight="bold")
    axis.text(0.5, 1.035, subtitle, transform=axis.transAxes, ha="center", va="bottom",
              fontsize=9.5)
    axis.text(-0.18, 1.18, letter, transform=axis.transAxes, ha="left", va="top",
              fontsize=12, fontweight="bold")
    for row in range(5):
        for column in range(5):
            # The original SI uses black annotations above 60% of its local
            # color range and white annotations below that value.
            color = "black" if norm(values[row, column]) > 0.6 else "white"
            axis.text(column, row, f"{values[row, column]:.2f}",
                      ha="center", va="center", color=color,
                      fontsize=7.5, fontweight="bold")
    bar = fig.colorbar(image, cax=color_axis)
    ticks = np.linspace(low, high, 5)
    bar.set_ticks(ticks)
    # 3 decimals avoid ambiguous duplicate tick labels for narrow ranges.
    bar.set_ticklabels([f"{v:.3f}" if high - low < 0.15 else f"{v:.2f}" for v in ticks])
    bar.ax.tick_params(length=2.3, width=0.6, labelsize=7.5, pad=2)
    bar.outline.set_linewidth(0.6)
    return axis, color_axis, {"vmin": low, "vmax": high, "origin": "upper",
                              "displayed_top_row": values[0].tolist(),
                              "displayed_bottom_row": values[-1].tolist()}

def assert_layout(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    checked = 0
    for axis in axes:
        texts = list(axis.texts) + [axis.xaxis.label, axis.yaxis.label]
        texts += axis.get_xticklabels() + axis.get_yticklabels()
        for text in texts:
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height:
                raise ValueError(f"Figure label outside canvas: {text.get_text()}")
            checked += 1
    return checked

def draw_figure(group, output, number, specs, nrows, ncols, height_mm):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, height_mm / 25.4))
    grid = fig.add_gridspec(nrows, ncols, left=0.065, right=0.925,
                           bottom=0.13 if nrows == 1 else 0.085,
                           top=0.84 if nrows == 1 else 0.92,
                           wspace=0.56 if ncols == 3 else 0.42, hspace=0.55)
    records, used, axes = [], [], []
    for index, spec in enumerate(specs):
        population, module, metric, title, subtitle = spec
        values, selected = matrix_for(group, population, module, metric)
        axis, cax, rendering = panel(fig, grid[index // ncols, index % ncols], values,
                                    title, subtitle, chr(ord("A") + index))
        axes.extend((axis, cax))
        record = {"figure": number, "panel": chr(ord("A") + index),
                  "population": population, "module": module, "metric": metric,
                  "colormap": "viridis", "normalization": "linear_per_panel_full_range",
                  **rendering}
        records.append(record)
        used.append(selected.assign(figure=number, panel=record["panel"]))
    labels_checked = assert_layout(fig, axes)
    return export(fig,number,output)

def plot_strength(number,output_dir=None):
    group=pd.read_csv(DATA/f"supp_figure_{number}"/"group_values.csv")
    apply_style()
    if number==13:
        specs=[("grid",m,"grid_rescale_factor",f"Grid Module {m}",r"$F_{\mathrm{R}}$") for m in (1,2,3)]
        specs += [("grid",m,"grid_phase_delta_x_cm",f"Grid Module {m}",r"$\Delta S_{\mathrm{phase}}$ (cm)") for m in (1,2,3)]
        return draw_figure(group,output_dir,number,specs,2,3,118.)
    specs=[("place",0,"place_field_length_delta_x_cm","Place cells",r"$\Delta L_{\mathrm{PF}}$ (cm)"),
           ("place",0,"peak_rate_hz","Place cells","Peak firing rate (Hz)")]
    return draw_figure(group,output_dir,number,specs,1,2,82.)
