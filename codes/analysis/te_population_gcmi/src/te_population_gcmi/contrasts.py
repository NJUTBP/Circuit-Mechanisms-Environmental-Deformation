"""Construction of independent-network transfer-entropy contrasts.

This module deliberately contains no estimator code.  It accepts a tidy table of
direction-specific TE estimates and turns it into the paired, network-level
quantities used for inference.  The hierarchy is always respected:

1. match P-to-G and G-to-P estimates within the same elementary unit;
2. average grid modules within each trajectory;
3. average trajectories within each network;
4. subtract the matched familiar baseline within network; and
5. form the prespecified state-by-axis difference-in-differences.

Cells, scalar pairs, segments, modules, and trajectories never become
independent observations in the returned primary contrast table.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd


MissingPolicy = Literal["raise", "drop"]
DuplicatePolicy = Literal["raise", "mean"]

DEFAULT_UNIT_COLUMNS = (
    "network_id",
    "trajectory_id",
    "module",
    "stage",
    "environment",
    "session",
    "state",
    "wall_axis",
)


def _as_list(columns: Sequence[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _ordered_unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], caller: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{caller}: missing required columns: {missing}")


def _validate_policy(value: str, allowed: set[str], name: str) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {{{choices}}}; got {value!r}")


def build_direction_difference(
    direction_table: pd.DataFrame,
    *,
    value_col: str = "te_bits",
    direction_col: str = "direction",
    unit_cols: Sequence[str] = DEFAULT_UNIT_COLUMNS,
    sensitivity_cols: Sequence[str] = (),
    p_to_g_labels: Sequence[str] = (
        "P_to_G",
        "P->G",
        "P→G",
        "place_to_grid",
        "place→grid",
    ),
    g_to_p_labels: Sequence[str] = (
        "G_to_P",
        "G->P",
        "G→P",
        "grid_to_place",
        "grid→place",
    ),
    missing: MissingPolicy = "raise",
    duplicates: DuplicatePolicy = "raise",
) -> pd.DataFrame:
    """Pair the two directions and compute ``D = T_PtoG - T_GtoP``.

    Parameters
    ----------
    direction_table:
        One row per elementary analysis unit and direction.  At minimum it must
        identify network, trajectory, grid module, environment, movement state,
        wall-axis class, direction, and TE value.
    unit_cols, sensitivity_cols:
        Together these columns uniquely identify an estimate before direction is
        added.  Every estimator setting that may change a TE value (for example
        lag, history, number of PCA components, conditioning set, or fold) must
        be supplied in ``sensitivity_cols``.
    missing:
        ``"raise"`` (default) prevents silent loss of a direction. ``"drop"``
        discards incomplete direction pairs and records the original row count
        in DataFrame attributes.
    duplicates:
        Duplicate estimates for the same unit/direction normally indicate an
        upstream schema error.  ``"mean"`` is available only for deliberately
        repeated estimates such as already-defined cross-validation folds.

    Returns
    -------
    pandas.DataFrame
        Tidy paired estimates with columns ``te_p_to_g``, ``te_g_to_p``, and
        ``D``.  The bounded directionality ratio is intentionally not computed.
    """

    _validate_policy(missing, {"raise", "drop"}, "missing")
    _validate_policy(duplicates, {"raise", "mean"}, "duplicates")
    keys = _ordered_unique([*_as_list(unit_cols), *_as_list(sensitivity_cols)])
    required = [*keys, direction_col, value_col]
    _require_columns(direction_table, required, "build_direction_difference")

    work = direction_table.loc[:, required].copy()
    if work[keys].isna().to_numpy().any():
        raise ValueError(
            "Pairing identifiers may not be missing; encode every sensitivity "
            "branch as an explicit level"
        )
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    p_labels = set(p_to_g_labels)
    g_labels = set(g_to_p_labels)
    overlap = p_labels.intersection(g_labels)
    if overlap:
        raise ValueError(f"Direction label sets overlap: {sorted(overlap)}")

    labels = work[direction_col]
    work["_canonical_direction"] = np.select(
        [labels.isin(p_labels), labels.isin(g_labels)],
        ["te_p_to_g", "te_g_to_p"],
        default="unknown",
    )
    unknown = sorted(work.loc[work["_canonical_direction"] == "unknown", direction_col].astype(str).unique())
    if unknown:
        raise ValueError(
            "Unrecognised direction labels. Pass explicit p_to_g_labels and "
            f"g_to_p_labels. Unknown values: {unknown}"
        )

    duplicate_mask = work.duplicated([*keys, "_canonical_direction"], keep=False)
    if duplicate_mask.any():
        if duplicates == "raise":
            example = work.loc[duplicate_mask, [*keys, direction_col]].head(8)
            raise ValueError(
                "Duplicate unit/direction estimates detected. Add the omitted "
                "setting to sensitivity_cols or aggregate deliberately.\n"
                f"Example rows:\n{example.to_string(index=False)}"
            )
        work = (
            work.groupby([*keys, "_canonical_direction"], dropna=False, as_index=False)[value_col]
            .mean()
        )

    wide = work.pivot(index=keys, columns="_canonical_direction", values=value_col).reset_index()
    wide.columns.name = None
    for column in ("te_p_to_g", "te_g_to_p"):
        if column not in wide:
            wide[column] = np.nan

    incomplete = wide[["te_p_to_g", "te_g_to_p"]].isna().any(axis=1)
    n_incomplete = int(incomplete.sum())
    if n_incomplete and missing == "raise":
        example = wide.loc[incomplete, keys].head(8)
        raise ValueError(
            f"{n_incomplete} elementary units lack one or both TE directions. "
            "Use missing='drop' only if this exclusion is scientifically justified.\n"
            f"Example units:\n{example.to_string(index=False)}"
        )
    if n_incomplete:
        wide = wide.loc[~incomplete].copy()

    wide["D"] = wide["te_p_to_g"] - wide["te_g_to_p"]
    wide.attrs["n_incomplete_direction_pairs"] = n_incomplete
    wide.attrs["direction_convention"] = "D = T_PtoG - T_GtoP"
    return wide.reset_index(drop=True)


def aggregate_to_network(
    paired_table: pd.DataFrame,
    *,
    value_cols: Sequence[str] = ("te_p_to_g", "te_g_to_p", "D"),
    network_col: str = "network_id",
    trajectory_col: str = "trajectory_id",
    module_col: str = "module",
    condition_cols: Sequence[str] = (
        "stage",
        "environment",
        "session",
        "state",
        "wall_axis",
    ),
    sensitivity_cols: Sequence[str] = (),
    expected_trajectories: int | None = None,
    expected_modules: int | None = None,
    missing: MissingPolicy = "raise",
) -> pd.DataFrame:
    """Average modules, then trajectories, to obtain one row per network/cell.

    Each module receives equal weight within a trajectory and each trajectory
    receives equal weight within a network.  With incomplete designs this differs
    from pooling all available module-trajectory rows; the staged mean prevents a
    trajectory with more available modules from receiving greater weight.

    When expected counts are supplied, the default ``missing='raise'`` rejects
    incomplete trajectories or networks.  With ``missing='drop'``, incomplete
    units are retained but count/range columns make the imbalance explicit.
    """

    _validate_policy(missing, {"raise", "drop"}, "missing")
    values = _as_list(value_cols)
    group_cols = _ordered_unique(
        [network_col, *_as_list(condition_cols), *_as_list(sensitivity_cols)]
    )
    required = [*group_cols, trajectory_col, module_col, *values]
    _require_columns(paired_table, required, "aggregate_to_network")

    work = paired_table.loc[:, required].copy()
    if work[[*group_cols, trajectory_col, module_col]].isna().to_numpy().any():
        raise ValueError("Aggregation identifiers may not be missing")
    for column in values:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    invalid_values = work[values].isna().any(axis=1)
    if invalid_values.any() and missing == "raise":
        raise ValueError(
            f"aggregate_to_network: {int(invalid_values.sum())} rows contain "
            "missing/non-numeric TE values"
        )
    work = work.loc[~invalid_values].copy()

    elementary_keys = [*group_cols, trajectory_col, module_col]
    duplicates = work.duplicated(elementary_keys, keep=False)
    if duplicates.any():
        example = work.loc[duplicates, elementary_keys].head(8)
        raise ValueError(
            "More than one paired TE row exists for a module/trajectory cell. "
            "Add omitted estimator settings to sensitivity_cols.\n"
            f"Example rows:\n{example.to_string(index=False)}"
        )

    trajectory_keys = [*group_cols, trajectory_col]
    module_counts = (
        work.groupby(trajectory_keys, dropna=False)[module_col]
        .nunique()
        .rename("n_modules")
        .reset_index()
    )
    if expected_modules is not None:
        incomplete_modules = module_counts["n_modules"] != expected_modules
        if incomplete_modules.any() and missing == "raise":
            example = module_counts.loc[incomplete_modules].head(8)
            raise ValueError(
                "Incomplete grid-module coverage within trajectories.\n"
                f"Expected {expected_modules}; examples:\n{example.to_string(index=False)}"
            )

    trajectory_means = (
        work.groupby(trajectory_keys, dropna=False, as_index=False)[values]
        .mean()
        .merge(module_counts, on=trajectory_keys, how="left", validate="one_to_one")
    )

    trajectory_counts = (
        trajectory_means.groupby(group_cols, dropna=False)[trajectory_col]
        .nunique()
        .rename("n_trajectories")
        .reset_index()
    )
    if expected_trajectories is not None:
        incomplete_trajectories = trajectory_counts["n_trajectories"] != expected_trajectories
        if incomplete_trajectories.any() and missing == "raise":
            example = trajectory_counts.loc[incomplete_trajectories].head(8)
            raise ValueError(
                "Incomplete trajectory coverage within networks.\n"
                f"Expected {expected_trajectories}; examples:\n{example.to_string(index=False)}"
            )

    network_values = trajectory_means.groupby(group_cols, dropna=False, as_index=False)[values].mean()
    module_summary = (
        trajectory_means.groupby(group_cols, dropna=False)["n_modules"]
        .agg(n_modules_min="min", n_modules_max="max")
        .reset_index()
    )
    result = (
        network_values.merge(trajectory_counts, on=group_cols, how="left", validate="one_to_one")
        .merge(module_summary, on=group_cols, how="left", validate="one_to_one")
    )
    result["complete_trajectory_design"] = (
        True
        if expected_trajectories is None
        else result["n_trajectories"].eq(expected_trajectories)
    )
    result["complete_module_design"] = (
        True
        if expected_modules is None
        else (
            result["n_modules_min"].eq(expected_modules)
            & result["n_modules_max"].eq(expected_modules)
        )
    )
    result["aggregation"] = "equal_modules_within_trajectory_then_equal_trajectories"
    return result


def compute_familiar_deltas(
    network_table: pd.DataFrame,
    *,
    value_cols: Sequence[str] = ("te_p_to_g", "te_g_to_p", "D"),
    network_col: str = "network_id",
    environment_col: str = "environment",
    state_col: str = "state",
    axis_col: str = "wall_axis",
    target_id_cols: Sequence[str] = ("stage", "session"),
    sensitivity_cols: Sequence[str] = (),
    baseline_environment: str = "familiar",
    target_environments: Sequence[str] | None = None,
    missing: MissingPolicy = "raise",
) -> pd.DataFrame:
    """Subtract a state- and wall-axis-matched familiar baseline within network.

    Familiar rows must already use the same *relative* wall-axis labels as the
    deformation rows (for example ``"deformed"`` and ``"orthogonal"``).  If an
    absolute x/y baseline must be re-used for different deformation orientations,
    relabel/duplicate it explicitly upstream; this function never guesses that
    mapping. ``target_id_cols`` (normally stage and session) identify repeated
    target conditions but are intentionally not baseline-matching keys, allowing
    the same prespecified familiar baseline to be shared across target sessions.
    """

    _validate_policy(missing, {"raise", "drop"}, "missing")
    values = _as_list(value_cols)
    target_ids = _as_list(target_id_cols)
    match_cols = _ordered_unique(
        [network_col, state_col, axis_col, *_as_list(sensitivity_cols)]
    )
    required = _ordered_unique([*match_cols, environment_col, *target_ids, *values])
    _require_columns(network_table, required, "compute_familiar_deltas")

    work = network_table.loc[:, required].copy()
    if work[[*match_cols, environment_col, *target_ids]].isna().to_numpy().any():
        raise ValueError("Baseline-matching identifiers may not be missing")
    if target_environments is not None:
        target_set = set(target_environments)
        available = set(work[environment_col].dropna().unique())
        unavailable = sorted(target_set - available)
        if unavailable:
            raise ValueError(f"Target environments not present: {unavailable}")
    else:
        target_set = set(work[environment_col].dropna().unique()) - {baseline_environment}

    baseline = work.loc[work[environment_col] == baseline_environment].copy()
    targets = work.loc[work[environment_col].isin(target_set)].copy()
    if baseline.empty:
        raise ValueError(f"No baseline rows found for {baseline_environment=!r}")
    if targets.empty:
        raise ValueError("No non-baseline target rows remain")

    duplicate_baseline = baseline.duplicated(match_cols, keep=False)
    if duplicate_baseline.any():
        example = baseline.loc[duplicate_baseline, [*match_cols, environment_col]].head(8)
        raise ValueError(
            "Familiar baseline is not unique for the requested matching keys. "
            "Add the omitted setting to sensitivity_cols.\n"
            f"Example rows:\n{example.to_string(index=False)}"
        )
    duplicate_targets = targets.duplicated(
        [*match_cols, environment_col, *target_ids], keep=False
    )
    if duplicate_targets.any():
        example = targets.loc[
            duplicate_targets, [*match_cols, environment_col, *target_ids]
        ].head(8)
        raise ValueError(
            "Target condition is not unique for the requested matching keys.\n"
            f"Example rows:\n{example.to_string(index=False)}"
        )

    baseline_names = {column: f"{column}_familiar" for column in values}
    baseline = baseline.loc[:, [*match_cols, *values]].rename(columns=baseline_names)
    merged = targets.merge(
        baseline,
        on=match_cols,
        how="left",
        validate="many_to_one",
        indicator="_baseline_merge",
    )
    absent = merged["_baseline_merge"] != "both"
    for column in baseline_names.values():
        absent |= merged[column].isna()
    absent |= merged[values].isna().any(axis=1)
    n_absent = int(absent.sum())
    if n_absent and missing == "raise":
        example = merged.loc[absent, [*match_cols, environment_col]].head(8)
        raise ValueError(
            f"{n_absent} target rows lack a complete matched familiar baseline.\n"
            f"Example rows:\n{example.to_string(index=False)}"
        )
    merged = merged.loc[~absent].drop(columns="_baseline_merge").copy()

    for column in values:
        merged[f"delta_{column}"] = merged[column] - merged[f"{column}_familiar"]
    merged.attrs["n_unmatched_or_missing"] = n_absent
    merged.attrs["delta_convention"] = "target environment minus matched familiar baseline"
    return merged.reset_index(drop=True)


def compute_primary_did(
    delta_table: pd.DataFrame,
    *,
    value_cols: Sequence[str] = ("delta_D", "delta_te_p_to_g", "delta_te_g_to_p"),
    network_col: str = "network_id",
    environment_col: str = "environment",
    state_col: str = "state",
    axis_col: str = "wall_axis",
    target_id_cols: Sequence[str] = ("stage", "session"),
    sensitivity_cols: Sequence[str] = (),
    approach_label: str = "approach",
    departure_label: str = "departure",
    deformed_axis_label: str = "deformed",
    orthogonal_axis_label: str = "orthogonal",
    missing: MissingPolicy = "raise",
) -> pd.DataFrame:
    """Compute the primary network-level state-by-axis contrast.

    For each supplied delta value ``X``, the contrast is

    ``C_X = (X[approach,deformed] - X[departure,deformed])``
    ``      - (X[approach,orthogonal] - X[departure,orthogonal])``.

    The primary output ``C`` corresponds to ``delta_D``.  Direction-specific
    outputs ``C_P_to_G`` and ``C_G_to_P`` retain the decomposition needed to
    determine which direction produced a change in relative directionality.
    """

    _validate_policy(missing, {"raise", "drop"}, "missing")
    values = _as_list(value_cols)
    target_ids = _as_list(target_id_cols)
    id_cols = _ordered_unique(
        [network_col, environment_col, *target_ids, *_as_list(sensitivity_cols)]
    )
    required = [*id_cols, state_col, axis_col, *values]
    _require_columns(delta_table, required, "compute_primary_did")

    wanted_states = {approach_label, departure_label}
    wanted_axes = {deformed_axis_label, orthogonal_axis_label}
    work = delta_table.loc[
        delta_table[state_col].isin(wanted_states)
        & delta_table[axis_col].isin(wanted_axes),
        required,
    ].copy()
    if work[[*id_cols, state_col, axis_col]].isna().to_numpy().any():
        raise ValueError("DID cell identifiers may not be missing")
    if work.empty:
        raise ValueError("No rows match the requested state and wall-axis labels")

    cell_keys = [*id_cols, state_col, axis_col]
    duplicates = work.duplicated(cell_keys, keep=False)
    if duplicates.any():
        example = work.loc[duplicates, cell_keys].head(8)
        raise ValueError(
            "More than one delta exists per network/state/axis cell. Add the "
            "omitted estimator setting to sensitivity_cols.\n"
            f"Example rows:\n{example.to_string(index=False)}"
        )

    labels = [
        (approach_label, deformed_axis_label),
        (departure_label, deformed_axis_label),
        (approach_label, orthogonal_axis_label),
        (departure_label, orthogonal_axis_label),
    ]
    result: pd.DataFrame | None = None
    output_names = {
        "delta_D": "C",
        "delta_te_p_to_g": "C_P_to_G",
        "delta_te_g_to_p": "C_G_to_P",
    }

    for value_col in values:
        pivot = work.pivot(index=id_cols, columns=[state_col, axis_col], values=value_col)
        missing_cells = [label for label in labels if label not in pivot.columns]
        if missing_cells:
            for label in missing_cells:
                pivot[label] = np.nan
        pivot = pivot.loc[:, labels]
        incomplete = pivot.isna().any(axis=1)
        if incomplete.any() and missing == "raise":
            example = pivot.index[incomplete].to_frame(index=False).head(8)
            raise ValueError(
                f"{int(incomplete.sum())} network/configuration rows lack one "
                f"of the four cells required for {value_col}.\n"
                f"Example rows:\n{example.to_string(index=False)}"
            )
        pivot = pivot.loc[~incomplete].copy()
        pivot.columns = [
            f"{value_col}__{state}__{axis}" for state, axis in pivot.columns
        ]
        pivot = pivot.reset_index()

        a_def = f"{value_col}__{approach_label}__{deformed_axis_label}"
        d_def = f"{value_col}__{departure_label}__{deformed_axis_label}"
        a_orth = f"{value_col}__{approach_label}__{orthogonal_axis_label}"
        d_orth = f"{value_col}__{departure_label}__{orthogonal_axis_label}"
        state_def_name = f"state_difference_deformed__{value_col}"
        state_orth_name = f"state_difference_orthogonal__{value_col}"
        contrast_name = output_names.get(value_col, f"C__{value_col}")
        pivot[state_def_name] = pivot[a_def] - pivot[d_def]
        pivot[state_orth_name] = pivot[a_orth] - pivot[d_orth]
        pivot[contrast_name] = pivot[state_def_name] - pivot[state_orth_name]

        if result is None:
            result = pivot
        else:
            result = result.merge(pivot, on=id_cols, how="inner", validate="one_to_one")

    assert result is not None
    if {"C", "C_P_to_G", "C_G_to_P"}.issubset(result.columns):
        result["direction_decomposition_error"] = (
            result["C"] - (result["C_P_to_G"] - result["C_G_to_P"])
        )
    result["contrast_definition"] = (
        "(approach-departure on deformed axis) - "
        "(approach-departure on orthogonal axis)"
    )
    return result.reset_index(drop=True)


def build_primary_contrast_table(
    direction_table: pd.DataFrame,
    *,
    sensitivity_cols: Sequence[str] = (),
    value_col: str = "te_bits",
    direction_col: str = "direction",
    unit_cols: Sequence[str] = DEFAULT_UNIT_COLUMNS,
    expected_trajectories: int | None = 2,
    expected_modules: int | None = 3,
    baseline_environment: str = "familiar",
    target_environments: Sequence[str] | None = None,
    missing: MissingPolicy = "raise",
) -> pd.DataFrame:
    """Run the full contrast-construction chain and return one row per network.

    This convenience API uses the package's standard column names and strict
    paired-design defaults.  Use the individual functions when labels or column
    names differ.  The returned table is ready for functions in
    :mod:`te_population_gcmi.statistics`; it must not be expanded back to cells,
    modules, trajectories, or segments for inference.
    """

    paired = build_direction_difference(
        direction_table,
        value_col=value_col,
        direction_col=direction_col,
        unit_cols=unit_cols,
        sensitivity_cols=sensitivity_cols,
        missing=missing,
    )
    network = aggregate_to_network(
        paired,
        sensitivity_cols=sensitivity_cols,
        expected_trajectories=expected_trajectories,
        expected_modules=expected_modules,
        missing=missing,
    )
    deltas = compute_familiar_deltas(
        network,
        sensitivity_cols=sensitivity_cols,
        baseline_environment=baseline_environment,
        target_environments=target_environments,
        missing=missing,
    )
    return compute_primary_did(
        deltas,
        sensitivity_cols=sensitivity_cols,
        missing=missing,
    )


__all__ = [
    "DEFAULT_UNIT_COLUMNS",
    "aggregate_to_network",
    "build_direction_difference",
    "build_primary_contrast_table",
    "compute_familiar_deltas",
    "compute_primary_did",
]
