"""Prepare matched v4 network curves for Figure-6-style learning-rate plot.

The historical single-network simulation is intentionally not mixed with v4.
All four displayed conditions (intact and three 0x pathways) come from the
same four independently initialized v4 networks in compressed_80.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT.parent / "run/outputs/learning_rates/analysis"
DEFAULT_OUTPUT = ROOT / "outputs/learning_rate_zero_control"
ENVIRONMENT = "compressed_80"
CONDITIONS = {
    "Intact": "W_bp_1p0x",
    "Eta_bp=0": "W_bp_0p0x",
    "Eta_pg=0": "W_pg_0p0x",
    "Eta_gp=0": "W_gp_0p0x",
}
INTACT_ALIASES = ("W_bp_1p0x", "W_pg_1p0x", "W_gp_1p0x")
METRICS = ("rescale_factor", "spatial_information_bits_per_spike")
NETWORKS = (0, 1, 2, 3)
SESSIONS = tuple(range(10))


def check_intact_aliases(source: pd.DataFrame) -> None:
    aliases = source.loc[source["condition"].isin(INTACT_ALIASES)].copy()
    aliases = aliases.loc[
        ((aliases["metric"] == "rescale_factor") & (aliases["population"] == "grid"))
        | (aliases["metric"] == "spatial_information_bits_per_spike")
    ]
    key = ["network_id", "session", "population", "module", "metric"]
    for measure in ("network_mean", "n_cells_finite", "n_cells_total"):
        pivot = aliases.pivot(index=key, columns="condition", values=measure)
        if pivot.shape[0] != 280 or not pivot.notna().all().all():
            raise ValueError(f"Incomplete 1x intact aliases for {measure}")
        if not np.allclose(pivot.max(axis=1), pivot.min(axis=1), rtol=0, atol=1e-12):
            raise ValueError(f"The three 1x intact aliases differ in {measure}")


def run(input_dir: Path, output_dir: Path) -> None:
    audit = json.loads((input_dir / "experiment_plan_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise ValueError("Learning-rate experiment plan audit did not pass")
    data = pd.read_csv(input_dir / "network_values_long.csv")
    source = data.loc[
        data["environment"].eq(ENVIRONMENT)
        & data["condition"].isin((*CONDITIONS.values(), *INTACT_ALIASES))
        & data["metric"].isin(METRICS)
    ].copy()
    check_intact_aliases(source)
    source = source.loc[source["condition"].isin(CONDITIONS.values())]
    source = source.loc[
        ((source["metric"] == "rescale_factor") & (source["population"] == "grid"))
        | (source["metric"] == "spatial_information_bits_per_spike")
    ].copy()
    source["condition_label"] = source["condition"].map({v: k for k, v in CONDITIONS.items()})
    if source["condition_label"].isna().any():
        raise ValueError("Unmapped condition")
    if not np.isfinite(source["network_mean"]).all():
        raise ValueError("Non-finite population/module network mean")
    if (source["n_cells_finite"] <= 0).any():
        raise ValueError("A population/module has zero finite cells")

    # Aggregate cells within each network first. Different population/module
    # valid counts are weights; networks themselves then receive equal weight.
    source["weighted_value"] = source["network_mean"] * source["n_cells_finite"]
    key = ["condition_label", "condition", "network_id", "session", "metric"]
    curves = source.groupby(key, sort=False, as_index=False).agg(
        weighted_sum=("weighted_value", "sum"),
        n_cells_finite=("n_cells_finite", "sum"),
        n_cells_total=("n_cells_total", "sum"),
        n_subgroups=("module", "size"),
    )
    curves["network_mean"] = curves["weighted_sum"] / curves["n_cells_finite"]
    curves = curves.drop(columns="weighted_sum")
    expected = len(CONDITIONS) * len(NETWORKS) * len(SESSIONS) * len(METRICS)
    if len(curves) != expected or curves.duplicated(key).any():
        raise ValueError(f"Expected {expected} unique network curves; got {len(curves)}")
    if not set(curves["network_id"]) == set(NETWORKS):
        raise ValueError("Expected four independent networks 000–003")
    if not set(curves["session"]) == set(SESSIONS):
        raise ValueError("Expected all sessions 0–9")
    for metric, expected_subgroups, expected_cells in (
        ("rescale_factor", 3, 300),
        ("spatial_information_bits_per_spike", 4, 400),
    ):
        subset = curves.loc[curves["metric"].eq(metric)]
        if not subset["n_subgroups"].eq(expected_subgroups).all():
            raise ValueError(f"Wrong subgroup coverage for {metric}")
        if not subset["n_cells_total"].eq(expected_cells).all():
            raise ValueError(f"Wrong modeled-cell total for {metric}")

    # All conditions share the same initial-exposure state within each network.
    initial = curves.loc[curves["session"].eq(0)]
    identity = initial.pivot(index=["network_id", "metric"],
                             columns="condition_label", values="network_mean")
    if not np.allclose(identity.max(axis=1), identity.min(axis=1), rtol=0, atol=1e-12):
        raise ValueError("Session-0 values differ among matched conditions")

    group = curves.groupby(["condition_label", "session", "metric"],
                           sort=False, as_index=False).agg(
        mean=("network_mean", "mean"),
        sd_networks=("network_mean", "std"),
        n_networks=("network_mean", "count"),
        minimum=("network_mean", "min"),
        maximum=("network_mean", "max"),
        min_valid_cells=("n_cells_finite", "min"),
        max_valid_cells=("n_cells_finite", "max"),
    )
    if not group["n_networks"].eq(4).all():
        raise ValueError("Every group/session/metric must contain four networks")
    group["t95_halfwidth"] = stats.t.ppf(0.975, group["n_networks"] - 1) * (
        group["sd_networks"] / np.sqrt(group["n_networks"])
    )
    curves = curves.sort_values(key)
    group = group.sort_values(["condition_label", "session", "metric"])
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / "network_curves.csv", index=False)
    group.to_csv(output_dir / "group_curves.csv", index=False)
    report = {
        "source": str((input_dir / "network_values_long.csv").resolve()),
        "environment": ENVIRONMENT,
        "conditions": CONDITIONS,
        "intact_aliases_checked_identical": list(INTACT_ALIASES),
        "networks": list(NETWORKS),
        "sessions": list(SESSIONS),
        "rescale": "Edge5-M; finite grid cells pooled within network across modules 1–3",
        "spatial_information": "Skaggs bits/spike; finite place and grid cells pooled within network by valid-cell count",
        "summary": "equal-weight mean over four independent network values; Student-t 95% CI saved, not required for thin-network-line plot",
        "n_network_rows": int(len(curves)),
        "n_group_rows": int(len(group)),
        "spatial_information_min_valid_cells": int(curves.loc[curves["metric"].eq("spatial_information_bits_per_spike"), "n_cells_finite"].min()),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(curves)} network values and {len(group)} group values to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
