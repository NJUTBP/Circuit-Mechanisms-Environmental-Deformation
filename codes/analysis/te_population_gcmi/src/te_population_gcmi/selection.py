"""Baseline-only selection of one shared TE embedding specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PARAMETER_COLUMNS = [
    "pca_components",
    "lag_frames",
    "source_history",
    "target_history",
]


@dataclass(frozen=True)
class SelectionResult:
    """A frozen primary setting and its complete baseline score table."""

    parameters: dict[str, int | str]
    scores: pd.DataFrame
    value_column: str


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Parameter-selection table is missing columns: {missing}")


def select_shared_parameters(
    baseline_scores: pd.DataFrame,
    *,
    value_column: str = "te_cv_bits",
    primary_conditioning: str = "behavior_conditioned",
    fixed_pca_components: int | None = None,
    expected_baseline_stage: str = "baseline",
    expected_network_ids: Iterable[str | int] | None = None,
    expected_states: Sequence[str] = ("approach", "departure"),
    expected_wall_axes: Sequence[str] = ("x", "y"),
) -> SelectionResult:
    """Select one setting using familiar-baseline scores only.

    Selection is deliberately shared by both directions and all grid modules.
    When ``fixed_pca_components`` is supplied, representation dimensionality is
    prespecified (10-D in the primary analysis) and only delay/history settings
    are selected; lower-dimensional PCA results remain sensitivity analyses.
    A candidate is eligible only if every expected network has a finite score
    for the complete module × state × wall-axis × direction design.  The two
    directions are then averaged within each state/axis cell, state/axis cells
    within module, modules within network, and finally networks.  Missing cells
    can therefore never improve a candidate's score by selective survival.

    The input must already contain familiar-baseline rows only.  If a ``stage``
    column is present, every value must match ``expected_baseline_stage``; this
    guard makes condition leakage difficult to introduce accidentally while
    allowing projects to use a different explicit baseline label.
    """

    needed = [
        "network_id",
        "module",
        "state",
        "wall_axis",
        "direction",
        "conditioning",
        value_column,
        *PARAMETER_COLUMNS,
    ]
    _require_columns(baseline_scores, needed)
    if "stage" in baseline_scores:
        stages = set(baseline_scores["stage"].dropna().astype(str))
        expected_stage = str(expected_baseline_stage)
        if stages != {expected_stage}:
            raise ValueError(
                "Primary TE parameters may be selected only from the declared "
                f"baseline stage {expected_stage!r}; received {sorted(stages)}"
            )

    data = baseline_scores.loc[
        baseline_scores["conditioning"].astype(str) == primary_conditioning
    ].copy()
    if fixed_pca_components is not None:
        if int(fixed_pca_components) < 1:
            raise ValueError("fixed_pca_components must be positive")
        data = data.loc[
            pd.to_numeric(data["pca_components"], errors="coerce")
            == int(fixed_pca_components)
        ].copy()
    if data.empty:
        raise ValueError(
            f"No rows for conditioning={primary_conditioning!r}"
        )

    expected_state_set = {str(value) for value in expected_states}
    expected_axis_set = {str(value) for value in expected_wall_axes}
    if not expected_state_set or len(expected_state_set) != len(expected_states):
        raise ValueError("expected_states must contain unique non-empty labels")
    if not expected_axis_set or len(expected_axis_set) != len(expected_wall_axes):
        raise ValueError("expected_wall_axes must contain unique non-empty labels")

    data["_network_key"] = data["network_id"].astype(str)
    data["module"] = data["module"].astype(str)
    data["state"] = data["state"].astype(str)
    data["wall_axis"] = data["wall_axis"].astype(str)
    data["direction"] = data["direction"].astype(str)
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    candidate_columns = ["conditioning", *PARAMETER_COLUMNS]
    if data[candidate_columns].isna().to_numpy().any():
        raise ValueError("Candidate parameter columns may not contain missing values")

    required_directions = {"place_to_grid", "grid_to_place"}
    directions = set(data["direction"])
    if directions != required_directions:
        raise ValueError(
            "Baseline selection requires exactly the two declared directions; found "
            f"{sorted(directions)}"
        )

    observed_networks = set(data["_network_key"])
    expected = (
        observed_networks
        if expected_network_ids is None
        else {str(value) for value in expected_network_ids}
    )
    if observed_networks != expected:
        raise ValueError(
            "Baseline selection network set differs from the expected set; "
            f"missing={sorted(expected - observed_networks)}, "
            f"unexpected={sorted(observed_networks - expected)}"
        )

    modules_by_network: dict[str, tuple[str, ...]] = {}
    for network_key, network_rows in data.groupby("_network_key", sort=True):
        states = set(network_rows["state"])
        axes = set(network_rows["wall_axis"])
        if states != expected_state_set:
            raise ValueError(
                f"Network {network_key!r} baseline states are incomplete: "
                f"expected {sorted(expected_state_set)}, found {sorted(states)}"
            )
        if axes != expected_axis_set:
            raise ValueError(
                f"Network {network_key!r} baseline wall axes are incomplete: "
                f"expected {sorted(expected_axis_set)}, found {sorted(axes)}"
            )
        modules = tuple(sorted(set(network_rows["module"])))
        if not modules:
            raise ValueError(f"Network {network_key!r} has no grid modules")
        modules_by_network[str(network_key)] = modules

    elementary_keys = [
        "_network_key",
        "module",
        "state",
        "wall_axis",
        "direction",
        *candidate_columns,
    ]
    duplicate = data.duplicated(elementary_keys, keep=False)
    if duplicate.any():
        example = data.loc[duplicate, elementary_keys].head(8)
        raise ValueError(
            "Duplicate baseline selection cells detected.\n"
            f"Examples:\n{example.to_string(index=False)}"
        )

    complete_candidates: list[pd.DataFrame] = []
    rejected_candidates = 0
    for _, candidate in data.groupby(candidate_columns, dropna=False, sort=False):
        candidate_complete = True
        retained_networks: list[pd.DataFrame] = []
        for network_key in sorted(expected):
            network_rows = candidate.loc[
                candidate["_network_key"].eq(network_key)
                & np.isfinite(candidate[value_column])
            ].copy()
            observed_cells = {
                tuple(row)
                for row in network_rows[
                    ["module", "state", "wall_axis", "direction"]
                ].itertuples(index=False, name=None)
            }
            expected_cells = {
                (module, state, axis, direction)
                for module in modules_by_network[network_key]
                for state in expected_state_set
                for axis in expected_axis_set
                for direction in required_directions
            }
            if observed_cells != expected_cells:
                candidate_complete = False
                break
            retained_networks.append(network_rows)
        if candidate_complete:
            complete_candidates.extend(retained_networks)
        else:
            rejected_candidates += 1

    if not complete_candidates:
        raise ValueError(
            "No candidate setting has finite scores for the complete "
            "network × module × state × wall-axis × direction design"
        )
    complete_data = pd.concat(complete_candidates, ignore_index=True)
    direction_mean = (
        complete_data.groupby(
            [
                "_network_key",
                "module",
                "state",
                "wall_axis",
                *candidate_columns,
                "direction",
            ],
            dropna=False,
        )[value_column]
        .mean()
        .unstack("direction")
        .reset_index()
    )
    direction_mean = direction_mean.dropna(
        subset=["place_to_grid", "grid_to_place"]
    )
    direction_mean["bidirectional_score"] = direction_mean[
        ["place_to_grid", "grid_to_place"]
    ].mean(axis=1)

    by_module = (
        direction_mean.groupby(
            ["_network_key", "module", *candidate_columns], dropna=False
        )["bidirectional_score"]
        .mean()
        .reset_index()
    )
    by_network = (
        by_module.groupby(
            ["_network_key", *candidate_columns], dropna=False
        )["bidirectional_score"]
        .mean()
        .reset_index()
    )

    summary = (
        by_network.groupby(candidate_columns, dropna=False)["bidirectional_score"]
        .agg(mean_score="mean", median_score="median", sd_score="std", n_networks="count")
        .reset_index()
    )
    summary = summary.loc[summary["n_networks"] == len(expected)].copy()
    summary["design_complete"] = True
    summary["n_state_axis_cells_per_module"] = (
        len(expected_state_set) * len(expected_axis_set)
    )
    summary["n_candidates_rejected_incomplete"] = rejected_candidates
    if summary.empty:
        raise ValueError(
            "No candidate setting has a complete score for every expected network"
        )

    # Deterministic parsimony tie-breaks: smaller histories, then shorter delay.
    summary = summary.sort_values(
        [
            "mean_score",
            "median_score",
            "source_history",
            "target_history",
            "lag_frames",
            "pca_components",
        ],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    winner = summary.iloc[0]
    parameters: dict[str, int | str] = {
        "conditioning": primary_conditioning,
        "pca_components": int(winner["pca_components"]),
        "lag_frames": int(winner["lag_frames"]),
        "source_history": int(winner["source_history"]),
        "target_history": int(winner["target_history"]),
    }
    return SelectionResult(parameters=parameters, scores=summary, value_column=value_column)
