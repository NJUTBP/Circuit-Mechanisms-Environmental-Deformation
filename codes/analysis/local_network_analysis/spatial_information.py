"""Audit three copied cluster analyses and aggregate familiarization spatial information.

The cell-level bits/spike values were calculated on the cluster from saved
occupancy-normalized rate maps. This script does not claim to recompute that
estimator without rate maps/occupancy. It independently rematches each cell to
its own familiar baseline, checks the cluster delta and network summaries,
and writes compact local tables. Original copied analyses remain read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT.parent / "run" / "outputs"
DEFAULT_OUTPUT = PROJECT / "outputs" / "spatial_information"
EXPERIMENTS = {
    "learning_rate": ("learning_rates", 72, 1584, 12),
    "place_cells": ("place_cell_count", 16, 416, 4),
    "alpha": ("grid_spacing", 20, 160, 2),
}
SI = "spatial_information_bits_per_spike"
DELTA = "delta_spatial_information_bits_per_spike"
CELL_KEYS = ["population", "module", "cell_id"]
RUN_KEYS = ["condition", "network_id", "stage", "environment", "session"]
NETWORK_KEYS = [
    "condition", "parameter", "network_id", "stage", "environment",
    "length_cm", "session", "population", "module",
]
GROUP_KEYS = [
    "condition", "parameter", "stage", "environment", "length_cm",
    "session", "population", "module", "metric",
]
CELL_COLUMNS = NETWORK_KEYS + ["cell_id", SI, f"{SI}_baseline", DELTA]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_copy(data_root: Path, experiment: str) -> tuple[dict, pd.DataFrame]:
    name, expected_shards, expected_qc, expected_png = EXPERIMENTS[experiment]
    root = data_root / name / "analysis"
    required = (
        "task_integrity_qc.csv", "condition_completeness.csv", "experiment_plan_audit.json",
        "network_values_long.csv", "group_summary_long.csv", "cell_files_manifest.csv",
        "source_file_manifest.csv", "pairing_audit.csv", "planned_network_contrasts.csv",
        "provenance.json",
    )
    absent = [item for item in required if not (root / item).is_file()]
    if absent:
        raise RuntimeError(f"{experiment}: missing exported files: {absent}")
    plan = json.loads((root / "experiment_plan_audit.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if plan.get("status") != "pass" or plan.get("issues"):
        raise RuntimeError(f"{experiment}: experiment plan audit did not pass: {plan}")
    qc = pd.read_csv(root / "task_integrity_qc.csv")
    completeness = pd.read_csv(root / "condition_completeness.csv")
    manifest = pd.read_csv(root / "cell_files_manifest.csv")
    network = pd.read_csv(root / "network_values_long.csv")
    group = pd.read_csv(root / "group_summary_long.csv")
    if len(qc) != expected_qc or not qc["included"].all():
        raise RuntimeError(f"{experiment}: QC count or inclusion differs from {expected_qc}")
    if qc.duplicated(RUN_KEYS).any():
        raise RuntimeError(f"{experiment}: duplicate logical QC runs")
    compared = qc.loc[qc.stage.ne("init")]
    if not compared["included_comparison"].all():
        raise RuntimeError(f"{experiment}: some comparison runs were excluded")
    if (completeness.n_analyzed_runs != completeness.n_expected_runs).any():
        raise RuntimeError(f"{experiment}: incomplete condition/session rows")
    if int(provenance.get("expected_task_count", -1)) != expected_qc:
        raise RuntimeError(f"{experiment}: provenance task count mismatch")
    if int(provenance.get("included_task_count", -1)) != expected_qc:
        raise RuntimeError(f"{experiment}: provenance included task count mismatch")
    if len(manifest) != expected_shards or manifest.path.duplicated().any():
        raise RuntimeError(f"{experiment}: expected {expected_shards} unique cell shards")
    if not {"condition_or_shard", "network_id", "path", "bytes", "sha256"}.issubset(manifest):
        raise RuntimeError(f"{experiment}: cell-file manifest has missing columns")
    actual_shards = list((root / "shards").glob("*__network_*"))
    if len(actual_shards) != expected_shards:
        raise RuntimeError(f"{experiment}: expected {expected_shards} shard directories")
    for row in manifest.itertuples(index=False):
        path = root / row.path
        if not path.is_file() or path.stat().st_size != int(row.bytes):
            raise RuntimeError(f"{experiment}: cell shard missing or wrong size: {path}")
        if sha256(path).lower() != str(row.sha256).lower():
            raise RuntimeError(f"{experiment}: cell shard SHA-256 mismatch: {path}")
    pngs = sorted((root / "figures").glob("*.png"))
    pdfs = sorted((root / "figures").glob("*.pdf"))
    if len(pngs) != expected_png or len(pdfs) != expected_png:
        raise RuntimeError(f"{experiment}: expected {expected_png} PNG/PDF preview pairs")
    if any(path.stat().st_size == 0 for path in pngs + pdfs):
        raise RuntimeError(f"{experiment}: empty preview figure")
    network_key = [
        "condition", "network_id", "environment", "session",
        "population", "module", "metric",
    ]
    if network.duplicated(network_key).any():
        raise RuntimeError(f"{experiment}: duplicate network summary keys")
    expected_network_rows = len(compared) * 96  # four cell groups x 24 metrics
    if len(network) != expected_network_rows:
        raise RuntimeError(
            f"{experiment}: {len(network)} network rows, expected {expected_network_rows}"
        )
    if group.duplicated([key for key in GROUP_KEYS if key != "stage"]).any():
        raise RuntimeError(f"{experiment}: duplicate group summary keys")
    si_network = network.loc[network.metric.eq(SI)]
    if si_network.empty or not np.isfinite(si_network.network_mean.to_numpy(float)).all():
        raise RuntimeError(f"{experiment}: missing/nonfinite network spatial information")
    report = {
        "experiment": experiment, "status": "pass",
        "analysis_source": str(root), "qc_rows": len(qc),
        "complete_comparison_runs": len(compared),
        "cell_shards_verified_sha256": len(manifest),
        "network_rows": len(network), "network_si_rows": len(si_network),
        "figure_png_pdf_pairs": len(pngs),
        "conditions": sorted(network.condition.unique().tolist()),
        "independent_network_ids": sorted(map(int, network.network_id.unique())),
        "unverifiable_after_copy": (
            "Original transient_activity.npz and rate-map/occupancy source files "
            "remain on the cluster; this audit verifies exported per-cell shards "
            "and their consistency, not the original simulation files."
        ),
    }
    return report, manifest


def _check_equal(label: str, actual: np.ndarray, recorded: np.ndarray) -> None:
    if not np.allclose(actual, recorded, rtol=1e-10, atol=1e-10, equal_nan=True):
        bad = int((~np.isclose(actual, recorded, rtol=1e-10, atol=1e-10, equal_nan=True)).sum())
        raise RuntimeError(f"{label}: {bad} rows disagree with cluster calculation")


def read_and_match_cells(source: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    selected = CELL_COLUMNS + ["trajectory_id"]
    for entry in manifest.itertuples(index=False):
        path = source / entry.path
        cells = pd.read_csv(path, usecols=selected)
        if cells.empty or cells.duplicated(RUN_KEYS + CELL_KEYS).any():
            raise RuntimeError(f"{path}: empty file or duplicate cell/run identity")
        if cells.condition.nunique() != 1 or cells.network_id.nunique() != 1:
            raise RuntimeError(f"{path}: mixed condition/network in one shard")
        if not cells.trajectory_id.eq(0).all():
            raise RuntimeError(f"{path}: unexpected nonzero trajectory")
        baseline = cells.loc[
            cells.stage.eq("baseline") & cells.environment.eq("fam_100")
            & cells.session.eq(0), CELL_KEYS + [SI]
        ]
        if baseline.empty or baseline.duplicated(CELL_KEYS).any():
            raise RuntimeError(f"{path}: missing or duplicated familiar baseline")
        merged = cells.merge(
            baseline.rename(columns={SI: "_matched_baseline_si"}),
            on=CELL_KEYS, how="left", validate="many_to_one",
        )
        _check_equal(
            str(path) + " baseline match",
            merged["_matched_baseline_si"].to_numpy(float),
            merged[f"{SI}_baseline"].to_numpy(float),
        )
        calculated_delta = (
            merged[SI].to_numpy(float) - merged["_matched_baseline_si"].to_numpy(float)
        )
        _check_equal(
            str(path) + " SI change",
            calculated_delta, merged[DELTA].to_numpy(float),
        )
        merged[DELTA] = calculated_delta
        merged[f"{SI}_baseline"] = merged["_matched_baseline_si"]
        frames.append(merged[CELL_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def network_values(cells: pd.DataFrame, exported_path: Path) -> pd.DataFrame:
    frame = cells.groupby(NETWORK_KEYS, dropna=False).agg(
        n_cells_total=("cell_id", "size"),
        spatial_information_bits_per_spike=(SI, "mean"),
        n_cells_finite_si=(SI, "count"),
        delta_spatial_information_bits_per_spike=(DELTA, "mean"),
        n_cells_finite_delta=(DELTA, "count"),
    ).reset_index()
    if frame[["n_cells_finite_si", "n_cells_finite_delta"]].eq(0).any().any():
        raise RuntimeError("At least one network/session has no finite spatial-information cells")
    long = pd.concat([
        frame[NETWORK_KEYS + ["n_cells_total", "n_cells_finite_si", SI]]
        .rename(columns={"n_cells_finite_si": "n_cells_finite", SI: "network_mean"})
        .assign(metric=SI),
        frame[NETWORK_KEYS + ["n_cells_total", "n_cells_finite_delta", DELTA]]
        .rename(columns={"n_cells_finite_delta": "n_cells_finite", DELTA: "network_mean"})
        .assign(metric=DELTA),
    ], ignore_index=True)
    exported = pd.read_csv(exported_path)
    exported = exported.loc[exported.metric.isin((SI, DELTA))]
    join_keys = [
        "condition", "parameter", "network_id", "environment",
        "length_cm", "session", "population", "module", "metric",
    ]
    compare = long.merge(
        exported[join_keys + ["network_mean", "n_cells_finite", "n_cells_total"]],
        on=join_keys, how="outer", validate="one_to_one",
        suffixes=("_local", "_cluster"), indicator=True,
    )
    if not compare["_merge"].eq("both").all() or len(compare) != len(long):
        raise RuntimeError("Local and cluster network spatial-information keys differ")
    _check_equal(
        "Network spatial information",
        compare.network_mean_local.to_numpy(float),
        compare.network_mean_cluster.to_numpy(float),
    )
    for field in ("n_cells_finite", "n_cells_total"):
        if not compare[f"{field}_local"].eq(compare[f"{field}_cluster"]).all():
            raise RuntimeError(f"Network spatial-information {field} disagrees")
    long["valid_fraction"] = long.n_cells_finite / long.n_cells_total
    return long.sort_values(join_keys).reset_index(drop=True)


def group_values(network: pd.DataFrame, exported_path: Path) -> pd.DataFrame:
    rows = []
    for key, part in network.groupby(GROUP_KEYS, dropna=False, sort=True):
        values = part.network_mean.to_numpy(float)
        values = values[np.isfinite(values)]
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
        rows.append(dict(zip(GROUP_KEYS, key)) | {
            "n_networks": n, "mean": mean, "sd_across_networks": sd,
            "ci95_low": mean - half, "ci95_high": mean + half,
        })
    frame = pd.DataFrame(rows)
    exported = pd.read_csv(exported_path)
    exported = exported.loc[exported.metric.isin((SI, DELTA))]
    join_keys = [key for key in GROUP_KEYS if key != "stage"]
    matched = frame.merge(
        exported[join_keys + ["mean", "ci95_low", "ci95_high", "n_networks"]],
        on=join_keys, how="outer", validate="one_to_one",
        suffixes=("_local", "_cluster"), indicator=True,
    )
    if len(matched) != len(frame) or not matched["_merge"].eq("both").all():
        raise RuntimeError("Local and cluster group spatial-information keys differ")
    for field in ("mean", "ci95_low", "ci95_high"):
        _check_equal(
            f"Group {field}",
            matched[f"{field}_local"].to_numpy(float),
            matched[f"{field}_cluster"].to_numpy(float),
        )
    if not matched.n_networks_local.eq(matched.n_networks_cluster).all():
        raise RuntimeError("Local and cluster group network counts differ")
    return frame


def _bh(p_values: np.ndarray) -> np.ndarray:
    result = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return result
    p = p_values[valid]
    order = np.argsort(p)
    sorted_q = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    restored = np.empty(len(p))
    restored[order] = np.minimum(sorted_q, 1.0)
    result[valid] = restored
    return result


def endpoint_contrasts(network: pd.DataFrame, pairing_path: Path,
                       experiment: str) -> pd.DataFrame:
    endpoint = network.loc[
        network.session.eq(9) & network.environment.isin(("compressed_80", "elongated_120"))
    ]
    pairing = pd.read_csv(pairing_path).set_index("condition")
    rows = []
    for (env, population, module, metric), part in endpoint.groupby(
        ["environment", "population", "module", "metric"], sort=True
    ):
        for condition, test in part.groupby("condition"):
            pathway = condition.rsplit("_", 1)[0] if experiment == "learning_rate" else "fixed_delta"
            reference = f"{pathway}_1p0x" if experiment == "learning_rate" else "npc100_reference"
            if condition == reference:
                continue
            ref = part.loc[part.condition.eq(reference)].set_index("network_id").network_mean
            joined = test.set_index("network_id").network_mean.to_frame("test").join(
                ref.rename("reference"), how="inner"
            ).dropna()
            if len(joined) < 2:
                continue
            verified = bool(pairing.loc[condition, "paired_allowed"])
            if not verified:
                raise RuntimeError(f"{experiment}: network pairing was not verified for {condition}")
            diff = joined["test"].to_numpy(float) - joined["reference"].to_numpy(float)
            n = len(diff)
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1))
            if np.allclose(diff, 0, atol=1e-12, rtol=0):
                t_value, p_value = 0.0, 1.0
            else:
                test_result = stats.ttest_1samp(diff, 0)
                t_value, p_value = float(test_result.statistic), float(test_result.pvalue)
            half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
            rows.append({
                "experiment": experiment, "pathway": pathway, "condition": condition,
                "reference_condition": reference, "environment": env,
                "population": population, "module": module, "metric": metric,
                "session": 9, "test": "two_sided_paired_network_t",
                "n_networks": n, "mean_paired_difference": mean,
                "ci95_low": mean - half, "ci95_high": mean + half,
                "t": t_value, "df": n - 1, "p_two_sided": p_value,
            })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["q_bh"] = np.nan
        family = ["experiment", "pathway", "environment", "population", "module", "metric"]
        for _, index in frame.groupby(family, dropna=False).groups.items():
            frame.loc[index, "q_bh"] = _bh(frame.loc[index, "p_two_sided"].to_numpy(float))
    return frame


def analyze_one(data_root: Path, output_root: Path, experiment: str,
                manifest: pd.DataFrame) -> dict:
    source = data_root / EXPERIMENTS[experiment][0] / "analysis"
    target = output_root / experiment
    target.mkdir(parents=True, exist_ok=True)
    cells = read_and_match_cells(source, manifest)
    cells.to_csv(target / "cell_spatial_information.csv.gz", index=False, compression="gzip")
    network = network_values(cells, source / "network_values_long.csv")
    network.to_csv(target / "network_spatial_information_long.csv", index=False)
    group = group_values(network, source / "group_summary_long.csv")
    group.to_csv(target / "group_spatial_information_long.csv", index=False)
    validity = network.groupby(GROUP_KEYS, dropna=False, sort=True).agg(
        n_networks=("network_id", "nunique"),
        mean_valid_fraction=("valid_fraction", "mean"),
        min_valid_fraction=("valid_fraction", "min"),
        max_valid_fraction=("valid_fraction", "max"),
        mean_finite_cells=("n_cells_finite", "mean"),
        mean_total_cells=("n_cells_total", "mean"),
    ).reset_index()
    validity.to_csv(target / "spatial_information_validity.csv", index=False)
    contrasts = endpoint_contrasts(network, source / "pairing_audit.csv", experiment)
    contrasts.to_csv(target / "endpoint_network_contrasts.csv", index=False)
    return {
        "cell_rows": len(cells), "network_rows": len(network),
        "group_rows": len(group), "endpoint_contrasts": len(contrasts),
        "place_nonfinite_si_rows": int(
            cells.loc[cells.population.eq("place"), SI].isna().sum()
        ),
        "place_nonfinite_baseline_matched_delta_rows": int(
            cells.loc[cells.population.eq("place"), DELTA].isna().sum()
        ),
        "network_n_min": int(group.n_networks.min()),
        "network_n_max": int(group.n_networks.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root, output_root = args.data_root.resolve(), args.output.resolve()
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_code_sha256": sha256(Path(__file__)),
        "spatial_information_source": (
            "Saved production per-cell bits/spike from 2.5-cm occupancy-normalized "
            "maps with Gaussian sigma=1.5 bins. Baseline matching, differences, "
            "network/group aggregation and endpoint tests are recomputed locally."
        ),
        "experiments": {},
    }
    manifests = {}
    for experiment in EXPERIMENTS:
        report, manifest = audit_copy(data_root, experiment)
        audit["experiments"][experiment] = report
        manifests[experiment] = manifest
    for experiment in ("learning_rate", "place_cells"):
        result = analyze_one(data_root, output_root, experiment, manifests[experiment])
        audit["experiments"][experiment]["spatial_information_analysis"] = result
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "copy_integrity_and_spatial_information_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for experiment, record in audit["experiments"].items():
        print(experiment, json.dumps(record.get("spatial_information_analysis", record),
                                      ensure_ascii=False))
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
