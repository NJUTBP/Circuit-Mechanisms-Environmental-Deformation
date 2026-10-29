"""Small-sample, network-level inference for transfer-entropy contrasts.

Only independent networks are resampled or sign-flipped.  Every public testing
function rejects duplicate rows for a network within a hypothesis, preventing
cells, modules, trajectories, or segments from being treated as the inferential
sample size.

The default missing-data rules are explicit:

* ordinary sign-flip tests and bootstrap intervals drop a network only within the
  affected hypothesis and report ``n_missing``;
* max-|t| familywise tests use complete networks across the entire family so that
  the same sign vector is applied to every sensitivity configuration;
* callers may choose ``missing='raise'`` to require a fully complete design.

No asymptotic t distribution is used.  With eight networks, the default sign-flip
test is exact (all 256 sign assignments).  Its validity requires independent
networks and exchangeability of the signs under a symmetric null distribution.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np
import pandas as pd


Alternative = Literal["two-sided", "greater", "less"]
ExactMode = Literal["auto"] | bool
MissingPolicy = Literal["drop", "raise"]
AdjustmentMethod = Literal["holm", "bh"]


def _as_list(columns: Sequence[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], caller: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{caller}: missing required columns: {missing}")


def _validate_choice(value: str, allowed: set[str], name: str) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {{{choices}}}; got {value!r}")


def _iter_groups(
    frame: pd.DataFrame, group_cols: Sequence[str]
) -> Iterator[tuple[tuple[object, ...], pd.DataFrame]]:
    columns = list(group_cols)
    if not columns:
        yield (), frame
        return
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, group in frame.groupby(grouper, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        yield key_tuple, group


def _group_fields(group_cols: Sequence[str], key: tuple[object, ...]) -> dict[str, object]:
    return dict(zip(group_cols, key, strict=True))


def _validate_network_rows(
    group: pd.DataFrame,
    *,
    network_col: str,
    value_col: str,
    missing: MissingPolicy,
    caller: str,
) -> tuple[np.ndarray, int, int]:
    if group[network_col].isna().any():
        raise ValueError(f"{caller}: network identifiers may not be missing")
    duplicated = group.duplicated(network_col, keep=False)
    if duplicated.any():
        ids = group.loc[duplicated, network_col].astype(str).unique()[:8]
        raise ValueError(
            f"{caller}: duplicate rows for independent networks {ids.tolist()}. "
            "Aggregate trajectories/modules before inference or add omitted "
            "hypothesis columns to group_cols."
        )

    numeric = pd.to_numeric(group[value_col], errors="coerce")
    n_total = int(group[network_col].nunique())
    n_missing = int(numeric.isna().sum())
    if n_missing and missing == "raise":
        missing_ids = group.loc[numeric.isna(), network_col].astype(str).tolist()[:8]
        raise ValueError(
            f"{caller}: {n_missing} network values are missing/non-numeric; "
            f"examples: {missing_ids}"
        )
    values = numeric.loc[numeric.notna()].to_numpy(dtype=float)
    return values, n_total, n_missing


def _student_t_1d(values: np.ndarray) -> float:
    n = values.size
    if n < 2:
        return float("nan")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if sd == 0.0:
        if mean == 0.0:
            return 0.0
        return float(np.copysign(np.inf, mean))
    return mean / (sd / np.sqrt(n))


def _student_t_rows(values: np.ndarray) -> np.ndarray:
    """Studentized means for an array shaped (draws, networks[, hypotheses])."""

    n = values.shape[1]
    if n < 2:
        return np.full(values.shape[:1] + values.shape[2:], np.nan, dtype=float)
    means = np.mean(values, axis=1)
    sds = np.std(values, axis=1, ddof=1)
    denominator = sds / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = means / denominator
    zero_both = (denominator == 0.0) & (means == 0.0)
    return np.where(zero_both, 0.0, stats)


def _resolve_exact(
    n_networks: int,
    exact: ExactMode,
    *,
    exact_max_networks: int,
    max_exact_permutations: int,
) -> bool:
    if exact not in (True, False, "auto"):
        raise ValueError("exact must be True, False, or 'auto'")
    use_exact = n_networks <= exact_max_networks if exact == "auto" else bool(exact)
    if use_exact and 2**n_networks > max_exact_permutations:
        raise ValueError(
            f"Exact sign enumeration requires {2**n_networks:,} assignments, "
            f"above max_exact_permutations={max_exact_permutations:,}. Use "
            "exact=False or increase the explicit guard."
        )
    return use_exact


def _sign_batches(
    n_networks: int,
    *,
    use_exact: bool,
    n_resamples: int,
    rng: np.random.Generator,
    batch_size: int,
) -> Iterator[np.ndarray]:
    if n_networks < 1:
        return
    if use_exact:
        total = 2**n_networks
        bit_positions = np.arange(n_networks, dtype=np.uint64)
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            codes = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = (codes >> bit_positions[None, :]) & np.uint64(1)
            yield np.where(bits == 1, 1.0, -1.0)
    else:
        if n_resamples < 1:
            raise ValueError("n_resamples must be positive for Monte Carlo tests")
        remaining = n_resamples
        while remaining:
            current = min(batch_size, remaining)
            yield rng.choice(np.array([-1.0, 1.0]), size=(current, n_networks))
            remaining -= current


def _extreme_mask(stats: np.ndarray, observed: float, alternative: Alternative) -> np.ndarray:
    if alternative == "two-sided":
        return np.abs(stats) >= abs(observed)
    if alternative == "greater":
        return stats >= observed
    return stats <= observed


def sign_flip_test(
    data: pd.DataFrame,
    *,
    value_col: str = "C",
    group_cols: Sequence[str] = (),
    network_col: str = "network_id",
    alternative: Alternative = "two-sided",
    statistic: Literal["t", "mean"] = "t",
    exact: ExactMode = "auto",
    exact_max_networks: int = 20,
    max_exact_permutations: int = 1_048_576,
    n_resamples: int = 100_000,
    random_state: int | np.random.Generator | None = 20260909,
    batch_size: int = 16_384,
    missing: MissingPolicy = "drop",
    small_n_threshold: int = 10,
) -> pd.DataFrame:
    """One-sample paired sign-flip inference across independent networks.

    The input must contain at most one row per network within each hypothesis
    defined by ``group_cols``.  Monte Carlo p-values use the finite-sampling-safe
    plus-one rule; exact p-values enumerate all ``2**n`` sign assignments.
    """

    _validate_choice(alternative, {"two-sided", "greater", "less"}, "alternative")
    _validate_choice(statistic, {"t", "mean"}, "statistic")
    _validate_choice(missing, {"drop", "raise"}, "missing")
    groups = _as_list(group_cols)
    _require_columns(data, [*groups, network_col, value_col], "sign_flip_test")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    rows: list[dict[str, object]] = []
    for key, group in _iter_groups(data, groups):
        values, n_total, n_missing = _validate_network_rows(
            group,
            network_col=network_col,
            value_col=value_col,
            missing=missing,
            caller="sign_flip_test",
        )
        n = values.size
        row = _group_fields(groups, key)
        row.update(
            {
                "value_col": value_col,
                "n_networks": n,
                "n_networks_total": n_total,
                "n_missing": n_missing,
                "estimate": float(np.mean(values)) if n else np.nan,
                "sd": float(np.std(values, ddof=1)) if n >= 2 else np.nan,
                "median": float(np.median(values)) if n else np.nan,
                "small_n": bool(n < small_n_threshold),
                "alternative": alternative,
                "statistic_name": statistic,
            }
        )
        if n < 2:
            row.update(
                {
                    "cohens_dz": np.nan,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "test_method": "not_tested",
                    "n_sign_assignments": 0,
                    "status": "insufficient_networks",
                }
            )
            rows.append(row)
            continue

        observed = _student_t_1d(values) if statistic == "t" else float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        mean = float(np.mean(values))
        cohens_dz = (
            0.0
            if sd == 0.0 and mean == 0.0
            else float(np.copysign(np.inf, mean))
            if sd == 0.0
            else mean / sd
        )
        use_exact = _resolve_exact(
            n,
            exact,
            exact_max_networks=exact_max_networks,
            max_exact_permutations=max_exact_permutations,
        )
        extreme = 0
        generated = 0
        for signs in _sign_batches(
            n,
            use_exact=use_exact,
            n_resamples=n_resamples,
            rng=rng,
            batch_size=batch_size,
        ):
            signed = signs * values[None, :]
            permutation_stats = (
                _student_t_rows(signed) if statistic == "t" else np.mean(signed, axis=1)
            )
            extreme += int(np.count_nonzero(_extreme_mask(permutation_stats, observed, alternative)))
            generated += int(signs.shape[0])
        p_value = extreme / generated if use_exact else (extreme + 1) / (generated + 1)
        row.update(
            {
                "statistic": observed,
                "cohens_dz": cohens_dz,
                "p_value": float(p_value),
                "test_method": "exact_sign_flip" if use_exact else "monte_carlo_sign_flip",
                "n_sign_assignments": generated,
                "status": "ok_small_n" if n < small_n_threshold else "ok",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def network_bootstrap_ci(
    data: pd.DataFrame,
    *,
    value_col: str = "C",
    group_cols: Sequence[str] = (),
    network_col: str = "network_id",
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    random_state: int | np.random.Generator | None = 20260909,
    batch_size: int = 4_096,
    missing: MissingPolicy = "drop",
    small_n_threshold: int = 10,
) -> pd.DataFrame:
    """Percentile confidence intervals from resampling whole networks.

    The interval is descriptive for very small ``n`` and does not replace the
    exact sign-flip test.  ``small_n`` and ``status`` expose that limitation in
    the returned tidy table.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    _validate_choice(missing, {"drop", "raise"}, "missing")
    groups = _as_list(group_cols)
    _require_columns(data, [*groups, network_col, value_col], "network_bootstrap_ci")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    alpha = 1.0 - confidence_level

    rows: list[dict[str, object]] = []
    for key, group in _iter_groups(data, groups):
        values, n_total, n_missing = _validate_network_rows(
            group,
            network_col=network_col,
            value_col=value_col,
            missing=missing,
            caller="network_bootstrap_ci",
        )
        n = values.size
        row = _group_fields(groups, key)
        row.update(
            {
                "value_col": value_col,
                "n_networks": n,
                "n_networks_total": n_total,
                "n_missing": n_missing,
                "estimate": float(np.mean(values)) if n else np.nan,
                "confidence_level": confidence_level,
                "n_bootstrap": n_resamples if n >= 2 else 0,
                "small_n": bool(n < small_n_threshold),
            }
        )
        if n < 2:
            row.update(
                {
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "bootstrap_se": np.nan,
                    "status": "insufficient_networks",
                }
            )
            rows.append(row)
            continue

        draws = np.empty(n_resamples, dtype=float)
        written = 0
        while written < n_resamples:
            current = min(batch_size, n_resamples - written)
            indices = rng.integers(0, n, size=(current, n))
            draws[written : written + current] = values[indices].mean(axis=1)
            written += current
        lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
        row.update(
            {
                "ci_low": float(lower),
                "ci_high": float(upper),
                "bootstrap_se": float(np.std(draws, ddof=1)),
                "status": "ok_small_n" if n < small_n_threshold else "ok",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm step-down familywise adjusted p-values; NaNs remain NaN."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if valid_positions.size == 0:
        return adjusted
    valid = values[valid_positions]
    if np.any((valid < 0.0) | (valid > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(valid, kind="stable")
    ordered = valid[order]
    multipliers = np.arange(ordered.size, 0, -1, dtype=float)
    ordered_adjusted = np.maximum.accumulate(ordered * multipliers)
    ordered_adjusted = np.minimum(ordered_adjusted, 1.0)
    restored = np.empty_like(ordered_adjusted)
    restored[order] = ordered_adjusted
    adjusted[valid_positions] = restored
    return adjusted


def benjamini_hochberg_adjust(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values; NaNs remain NaN."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if valid_positions.size == 0:
        return adjusted
    valid = values[valid_positions]
    if np.any((valid < 0.0) | (valid > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(valid, kind="stable")
    ordered = valid[order]
    ranks = np.arange(1, ordered.size + 1, dtype=float)
    ordered_adjusted = ordered * ordered.size / ranks
    ordered_adjusted = np.minimum.accumulate(ordered_adjusted[::-1])[::-1]
    ordered_adjusted = np.minimum(ordered_adjusted, 1.0)
    restored = np.empty_like(ordered_adjusted)
    restored[order] = ordered_adjusted
    adjusted[valid_positions] = restored
    return adjusted


def adjust_pvalues(
    table: pd.DataFrame,
    *,
    p_col: str = "p_value",
    method: AdjustmentMethod = "holm",
    family_cols: Sequence[str] = (),
    output_col: str | None = None,
) -> pd.DataFrame:
    """Add Holm or BH adjusted p-values within prespecified test families."""

    _validate_choice(method, {"holm", "bh"}, "method")
    families = _as_list(family_cols)
    _require_columns(table, [*families, p_col], "adjust_pvalues")
    output = output_col or f"p_{method}"
    result = table.copy()
    result[output] = np.nan
    adjuster = holm_adjust if method == "holm" else benjamini_hochberg_adjust
    for _, group in _iter_groups(result, families):
        result.loc[group.index, output] = adjuster(group[p_col].to_numpy(dtype=float))
    return result


def max_t_sign_flip(
    data: pd.DataFrame,
    *,
    value_col: str = "C",
    hypothesis_cols: Sequence[str],
    family_cols: Sequence[str] = (),
    network_col: str = "network_id",
    exact: ExactMode = "auto",
    exact_max_networks: int = 20,
    max_exact_permutations: int = 1_048_576,
    n_resamples: int = 100_000,
    random_state: int | np.random.Generator | None = 20260909,
    batch_size: int = 4_096,
    missing: Literal["complete_case", "raise"] = "complete_case",
    small_n_threshold: int = 10,
) -> pd.DataFrame:
    """Shared-sign max-|t| familywise correction across sensitivity settings.

    ``hypothesis_cols`` identify every comparison included in the maximum (for
    example environment, lag, history, PCA dimension, and conditioning model).
    ``family_cols`` split independent, prespecified families and must not overlap
    ``hypothesis_cols``.  Within a family, one sign vector is shared across all
    hypotheses on every permutation.

    The default complete-case policy drops a network from the *entire family* if
    any configuration is missing.  This maintains the common randomization set
    required for a coherent max statistic.  The number dropped is returned.
    """

    _validate_choice(missing, {"complete_case", "raise"}, "missing")
    hypotheses = _as_list(hypothesis_cols)
    families = _as_list(family_cols)
    if not hypotheses:
        raise ValueError("hypothesis_cols must contain at least one column")
    overlap = set(hypotheses).intersection(families)
    if overlap:
        raise ValueError(f"family_cols and hypothesis_cols overlap: {sorted(overlap)}")
    required = [*families, *hypotheses, network_col, value_col]
    _require_columns(data, required, "max_t_sign_flip")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    rows: list[dict[str, object]] = []
    for family_key, family in _iter_groups(data, families):
        if family[network_col].isna().any():
            raise ValueError("max_t_sign_flip: network identifiers may not be missing")
        cell_keys = [network_col, *hypotheses]
        duplicates = family.duplicated(cell_keys, keep=False)
        if duplicates.any():
            example = family.loc[duplicates, cell_keys].head(8)
            raise ValueError(
                "Duplicate network/hypothesis rows detected. Aggregate nested "
                "observations or add omitted settings to hypothesis_cols.\n"
                f"Example rows:\n{example.to_string(index=False)}"
            )

        family_work = family.loc[:, [*cell_keys, value_col]].copy()
        if family_work[hypotheses].isna().to_numpy().any():
            raise ValueError(
                "max_t_sign_flip: hypothesis identifiers may not be missing; "
                "encode an explicit level instead"
            )
        family_work[value_col] = pd.to_numeric(family_work[value_col], errors="coerce")
        wide = family_work.pivot(index=network_col, columns=hypotheses, values=value_col)
        n_total = int(wide.shape[0])
        incomplete_network = wide.isna().any(axis=1)
        n_dropped = int(incomplete_network.sum())
        if n_dropped and missing == "raise":
            ids = wide.index[incomplete_network].astype(str).tolist()[:8]
            raise ValueError(
                f"max_t_sign_flip: {n_dropped} networks are incomplete across "
                f"the correction family; examples: {ids}"
            )
        wide = wide.loc[~incomplete_network]
        values = wide.to_numpy(dtype=float)
        n, m = values.shape
        family_fields = _group_fields(families, family_key)

        if n >= 2:
            observed = np.array([_student_t_1d(values[:, j]) for j in range(m)])
            estimates = np.mean(values, axis=0)
            sds = np.std(values, axis=0, ddof=1)
            use_exact = _resolve_exact(
                n,
                exact,
                exact_max_networks=exact_max_networks,
                max_exact_permutations=max_exact_permutations,
            )
            raw_extreme = np.zeros(m, dtype=np.int64)
            max_extreme = np.zeros(m, dtype=np.int64)
            generated = 0
            for signs in _sign_batches(
                n,
                use_exact=use_exact,
                n_resamples=n_resamples,
                rng=rng,
                batch_size=batch_size,
            ):
                signed = signs[:, :, None] * values[None, :, :]
                permutation_t = _student_t_rows(signed)
                absolute_t = np.abs(permutation_t)
                max_absolute_t = np.max(absolute_t, axis=1)
                raw_extreme += np.count_nonzero(
                    absolute_t >= np.abs(observed)[None, :], axis=0
                )
                max_extreme += np.count_nonzero(
                    max_absolute_t[:, None] >= np.abs(observed)[None, :], axis=0
                )
                generated += int(signs.shape[0])
            if use_exact:
                raw_p = raw_extreme / generated
                adjusted_p = max_extreme / generated
            else:
                raw_p = (raw_extreme + 1) / (generated + 1)
                adjusted_p = (max_extreme + 1) / (generated + 1)
            method = "exact_shared_sign_max_t" if use_exact else "monte_carlo_shared_sign_max_t"
            status = "ok_small_n" if n < small_n_threshold else "ok"
        else:
            observed = np.full(m, np.nan)
            estimates = np.mean(values, axis=0) if n else np.full(m, np.nan)
            sds = np.full(m, np.nan)
            raw_p = np.full(m, np.nan)
            adjusted_p = np.full(m, np.nan)
            generated = 0
            method = "not_tested"
            status = "insufficient_complete_networks"

        columns = list(wide.columns)
        for j, column_key in enumerate(columns):
            hypothesis_key = column_key if isinstance(column_key, tuple) else (column_key,)
            row = {**family_fields, **_group_fields(hypotheses, hypothesis_key)}
            row.update(
                {
                    "value_col": value_col,
                    "n_networks": n,
                    "n_networks_total": n_total,
                    "n_networks_dropped_familywise": n_dropped,
                    "n_hypotheses_in_family": m,
                    "estimate": float(estimates[j]),
                    "sd": float(sds[j]),
                    "t_statistic": float(observed[j]),
                    "p_value": float(raw_p[j]),
                    "p_max_t_fwer": float(adjusted_p[j]),
                    "test_method": method,
                    "n_sign_assignments": generated,
                    "small_n": bool(n < small_n_threshold),
                    "missing_policy": missing,
                    "status": status,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def network_inference_table(
    data: pd.DataFrame,
    *,
    value_col: str = "C",
    hypothesis_cols: Sequence[str] = ("environment",),
    family_cols: Sequence[str] = (),
    network_col: str = "network_id",
    adjustment: AdjustmentMethod = "holm",
    confidence_level: float = 0.95,
    exact_max_networks: int = 20,
    sign_resamples: int = 100_000,
    bootstrap_resamples: int = 10_000,
    random_state: int | None = 20260909,
    missing: MissingPolicy = "drop",
) -> pd.DataFrame:
    """Combine sign-flip inference, network bootstrap CIs, and p adjustment.

    This convenience function is intended for the small prespecified primary
    family (for example compressed and elongated deformation contrasts).  Use
    :func:`max_t_sign_flip` for the larger lag/history/PCA sensitivity family.
    """

    hypotheses = _as_list(hypothesis_cols)
    families = _as_list(family_cols)
    group_cols = list(dict.fromkeys([*families, *hypotheses]))
    tests = sign_flip_test(
        data,
        value_col=value_col,
        group_cols=group_cols,
        network_col=network_col,
        exact_max_networks=exact_max_networks,
        n_resamples=sign_resamples,
        random_state=random_state,
        missing=missing,
    )
    intervals = network_bootstrap_ci(
        data,
        value_col=value_col,
        group_cols=group_cols,
        network_col=network_col,
        confidence_level=confidence_level,
        n_resamples=bootstrap_resamples,
        random_state=random_state,
        missing=missing,
    )
    interval_columns = [
        *group_cols,
        "ci_low",
        "ci_high",
        "bootstrap_se",
        "n_bootstrap",
    ]
    if group_cols:
        merged = tests.merge(
            intervals.loc[:, interval_columns],
            on=group_cols,
            how="left",
            validate="one_to_one",
        )
    else:
        if len(tests) != 1 or len(intervals) != 1:
            raise RuntimeError("Ungrouped inference unexpectedly produced multiple rows")
        merged = tests.copy()
        for column in interval_columns:
            merged[column] = intervals.iloc[0][column]
    return adjust_pvalues(
        merged,
        p_col="p_value",
        method=adjustment,
        family_cols=families,
    )


__all__ = [
    "adjust_pvalues",
    "benjamini_hochberg_adjust",
    "holm_adjust",
    "max_t_sign_flip",
    "network_bootstrap_ci",
    "network_inference_table",
    "sign_flip_test",
]
