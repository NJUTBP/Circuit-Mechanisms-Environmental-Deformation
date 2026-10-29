"""Configuration loading and validation for the TE analysis.

The validator deliberately checks the scientific contract, not only JSON
syntax.  Extra keys are retained so that provenance annotations can be added
without changing this package, while all fields used by the core pipeline are
type- and range-checked here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping


class ConfigError(ValueError):
    """Raised when a configuration violates the analysis contract."""


_SECTIONS = (
    "data",
    "baseline",
    "segmentation",
    "representation",
    "conditioning",
    "te",
    "cross_trajectory_null",
    "statistics",
    "execution",
)


def _apply_defaults(raw: dict[str, Any]) -> None:
    """Materialize every computational default in the resolved configuration."""

    data = _mapping(raw["data"], "data")
    data.setdefault("allow_nonfinite", False)
    data.setdefault("dt_relative_tolerance", 1e-6)
    data.setdefault("minimum_trajectories_per_group", 2)

    segmentation = _mapping(raw["segmentation"], "segmentation")
    segmentation.setdefault("minimum_monotonic_fraction", 0.0)
    segmentation.setdefault("minimum_path_length_m", 0.0)
    segmentation.setdefault("wall_tie_tolerance_m", 1e-9)

    te = _mapping(raw["te"], "te")
    te.setdefault("autocorrelation_max_lag", 100)
    te.setdefault("effect_value", "te_raw_bits")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a JSON object")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required setting {path}.{key}")
    return mapping[key]


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive_minimum: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be a finite number")
    if minimum is not None:
        invalid = result < minimum if inclusive_minimum else result <= minimum
        if invalid:
            comparator = ">=" if inclusive_minimum else ">"
            raise ConfigError(f"{path} must be {comparator} {minimum}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{path} must be <= {maximum}")
    return result


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _positive_integer_list(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list")
    result = [_integer(item, f"{path}[{i}]", minimum=1) for i, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise ConfigError(f"{path} must not contain duplicates")
    if result != sorted(result):
        raise ConfigError(f"{path} must be in ascending order")
    return result


def _array_key(value: Any, path: str) -> str:
    """Validate an NPZ key specification.

    String specifications mean that the stored array is already time-major.
    A mapping may be used to declare another orientation explicitly, e.g.
    ``{"key": "place", "time_axis": 1}``.  The loader never infers an axis.
    """

    if isinstance(value, str):
        return _string(value, path)
    spec = _mapping(value, path)
    key = _string(_required(spec, "key", path), f"{path}.key")
    time_axis = _integer(_required(spec, "time_axis", path), f"{path}.time_axis")
    if time_axis not in (0, 1):
        raise ConfigError(f"{path}.time_axis must be 0 or 1")
    return key


def _validate_data(section: dict[str, Any]) -> None:
    _string(_required(section, "manifest_path", "data"), "data.manifest_path")
    _number(
        _required(section, "expected_dt_s", "data"),
        "data.expected_dt_s",
        minimum=0.0,
        inclusive_minimum=False,
    )
    keys = _mapping(_required(section, "array_keys", "data"), "data.array_keys")
    used_keys: list[str] = []
    for logical in ("positions", "place"):
        used_keys.append(
            _array_key(
                _required(keys, logical, "data.array_keys"),
                f"data.array_keys.{logical}",
            )
        )
    modules = _mapping(
        _required(keys, "grid_modules", "data.array_keys"),
        "data.array_keys.grid_modules",
    )
    if not modules:
        raise ConfigError("data.array_keys.grid_modules must contain at least one module")
    for module, specification in modules.items():
        _string(module, "data.array_keys.grid_modules key")
        used_keys.append(
            _array_key(specification, f"data.array_keys.grid_modules.{module}")
        )
    for optional in ("bvc", "speed", "heading", "time", "dt_s"):
        if optional in keys:
            if optional == "dt_s":
                used_keys.append(_string(keys[optional], "data.array_keys.dt_s"))
            else:
                used_keys.append(
                    _array_key(keys[optional], f"data.array_keys.{optional}")
                )
    if len(set(used_keys)) != len(used_keys):
        raise ConfigError("Every data.array_keys entry must refer to a distinct NPZ key")
    require_bvc = section.get("require_bvc", False)
    _bool(require_bvc, "data.require_bvc")
    if require_bvc and "bvc" not in keys:
        raise ConfigError("data.require_bvc=true requires data.array_keys.bvc")
    if "allow_nonfinite" in section:
        _bool(section["allow_nonfinite"], "data.allow_nonfinite")
    if "dt_relative_tolerance" in section:
        _number(
            section["dt_relative_tolerance"],
            "data.dt_relative_tolerance",
            minimum=0.0,
        )
    if "minimum_trajectories_per_group" in section:
        _integer(
            section["minimum_trajectories_per_group"],
            "data.minimum_trajectories_per_group",
            minimum=2,
        )
    if "allow_single_trajectory_unconditioned" in section:
        _bool(
            section["allow_single_trajectory_unconditioned"],
            "data.allow_single_trajectory_unconditioned",
        )


def _validate_baseline(section: dict[str, Any]) -> None:
    _string(_required(section, "stage", "baseline"), "baseline.stage")
    _string(_required(section, "environment", "baseline"), "baseline.environment")
    session = _required(section, "session", "baseline")
    if isinstance(session, bool) or not isinstance(session, (str, int)):
        raise ConfigError("baseline.session must be a string or integer")


def _validate_segmentation(section: dict[str, Any]) -> None:
    source = _string(_required(section, "source", "segmentation"), "segmentation.source")
    if source != "classify":
        raise ConfigError("segmentation.source must currently be 'classify'")
    _number(
        _required(section, "boundary_zone_m", "segmentation"),
        "segmentation.boundary_zone_m",
        minimum=0.0,
        inclusive_minimum=False,
    )
    _number(
        _required(section, "centre_fraction", "segmentation"),
        "segmentation.centre_fraction",
        minimum=0.0,
        maximum=1.0,
        inclusive_minimum=False,
    )
    minimum = _integer(
        _required(section, "minimum_segment_frames", "segmentation"),
        "segmentation.minimum_segment_frames",
        minimum=2,
    )
    maximum = section.get("maximum_segment_frames")
    if maximum is not None and _integer(
        maximum, "segmentation.maximum_segment_frames", minimum=minimum
    ) < minimum:
        raise ConfigError(
            "segmentation.maximum_segment_frames must not be smaller than the minimum"
        )
    if "minimum_monotonic_fraction" in section:
        _number(
            section["minimum_monotonic_fraction"],
            "segmentation.minimum_monotonic_fraction",
            minimum=0.0,
            maximum=1.0,
        )
    if "wall_tie_tolerance_m" in section:
        _number(
            section["wall_tie_tolerance_m"],
            "segmentation.wall_tie_tolerance_m",
            minimum=0.0,
        )


    if "centre_anchor_extension" in section:
        spec = section["centre_anchor_extension"]
        if not isinstance(spec, dict) or spec.get("mode") != "entry_exit_arclength":
            raise ConfigError("centre_anchor_extension requires mode entry_exit_arclength")
        for key in ("base_range_m", "gap_compensation", "reference_length_m"):
            _number(_required(spec, key, "centre_anchor_extension"),
                    f"centre_anchor_extension.{key}", minimum=0.0)
        allowed = spec.get("allowed_widths_m")
        if not isinstance(allowed, list) or not allowed:
            raise ConfigError("centre_anchor_extension.allowed_widths_m must be nonempty")
        for value in allowed:
            _number(value, "centre_anchor_extension.allowed_widths_m", minimum=0.0,
                    inclusive_minimum=False)


def _validate_representation(section: dict[str, Any]) -> None:
    primary = _integer(
        _required(section, "primary_components", "representation"),
        "representation.primary_components",
        minimum=1,
    )
    sensitivity = section.get("sensitivity_components", [])
    if not isinstance(sensitivity, list):
        raise ConfigError("representation.sensitivity_components must be a list")
    parsed = [
        _integer(v, f"representation.sensitivity_components[{i}]", minimum=1)
        for i, v in enumerate(sensitivity)
    ]
    if len(set(parsed)) != len(parsed):
        raise ConfigError("representation.sensitivity_components contains duplicates")
    if primary in parsed:
        raise ConfigError(
            "representation.sensitivity_components must not repeat primary_components"
        )
    maximum_samples = _required(
        section, "pca_fit_max_samples", "representation"
    )
    if maximum_samples is not None:
        _integer(
            maximum_samples,
            "representation.pca_fit_max_samples",
            minimum=max([primary, *parsed], default=primary) + 1,
        )


def _validate_conditioning(section: dict[str, Any]) -> None:
    primary = _string(_required(section, "primary", "conditioning"), "conditioning.primary")
    branches = _required(section, "branches", "conditioning")
    if not isinstance(branches, list) or not branches:
        raise ConfigError("conditioning.branches must be a non-empty list")
    branches = [_string(v, f"conditioning.branches[{i}]") for i, v in enumerate(branches)]
    if len(set(branches)) != len(branches):
        raise ConfigError("conditioning.branches contains duplicates")
    allowed = {"unconditioned", "behavior_conditioned", "behavior_bvc_conditioned"}
    unknown = set(branches).difference(allowed)
    if unknown:
        raise ConfigError(f"Unknown conditioning branches: {sorted(unknown)}")
    if primary not in branches:
        raise ConfigError("conditioning.primary must occur in conditioning.branches")
    grid = _required(section, "position_rbf_grid", "conditioning")
    if isinstance(grid, int) and not isinstance(grid, bool):
        _integer(grid, "conditioning.position_rbf_grid", minimum=2)
    elif isinstance(grid, list) and len(grid) == 2:
        for i, value in enumerate(grid):
            _integer(value, f"conditioning.position_rbf_grid[{i}]", minimum=2)
    else:
        raise ConfigError("conditioning.position_rbf_grid must be an integer or [nx, ny]")
    _number(
        _required(section, "position_rbf_width", "conditioning"),
        "conditioning.position_rbf_width",
        minimum=0.0,
        inclusive_minimum=False,
    )
    _integer(
        _required(section, "speed_polynomial_degree", "conditioning"),
        "conditioning.speed_polynomial_degree",
        minimum=1,
    )
    _number(
        _required(section, "ridge", "conditioning"),
        "conditioning.ridge",
        minimum=0.0,
    )
    include_bvc = _bool(
        _required(section, "include_bvc_sensitivity", "conditioning"),
        "conditioning.include_bvc_sensitivity",
    )
    bvc_components = _integer(
        _required(section, "bvc_components", "conditioning"),
        "conditioning.bvc_components",
        minimum=1,
    )
    if include_bvc and bvc_components < 1:  # pragma: no cover - guarded above
        raise ConfigError("conditioning.bvc_components must be positive")
    has_bvc_branch = "behavior_bvc_conditioned" in branches
    if include_bvc != has_bvc_branch:
        raise ConfigError(
            "conditioning.include_bvc_sensitivity must be true exactly when "
            "conditioning.branches contains 'behavior_bvc_conditioned'"
        )


def _validate_te(section: dict[str, Any]) -> None:
    estimator = _string(
        _required(section, "primary_estimator", "te"), "te.primary_estimator"
    )
    if estimator != "gaussian_copula":
        raise ConfigError("te.primary_estimator must be 'gaussian_copula'")
    effect_value = _string(
        _required(section, "effect_value", "te"), "te.effect_value"
    )
    if effect_value not in {"te_raw_bits", "te_bias_corrected_bits"}:
        raise ConfigError(
            "te.effect_value must be te_raw_bits or te_bias_corrected_bits"
        )
    _bool(_required(section, "compute_cv_score", "te"), "te.compute_cv_score")
    _positive_integer_list(_required(section, "lag_frames", "te"), "te.lag_frames")
    _positive_integer_list(
        _required(section, "source_history_frames", "te"),
        "te.source_history_frames",
    )
    _positive_integer_list(
        _required(section, "target_history_frames", "te"),
        "te.target_history_frames",
    )
    _number(
        _required(section, "covariance_shrinkage", "te"),
        "te.covariance_shrinkage",
        minimum=0.0,
        maximum=1.0,
    )
    _number(
        _required(section, "regression_ridge", "te"),
        "te.regression_ridge",
        minimum=0.0,
    )
    for key in ("bias_shuffles", "sensitivity_bias_shuffles"):
        _integer(_required(section, key, "te"), f"te.{key}", minimum=0)
    _integer(
        _required(section, "minimum_shift_frames", "te"),
        "te.minimum_shift_frames",
        minimum=1,
    )
    _integer(
        _required(section, "minimum_pool_samples", "te"),
        "te.minimum_pool_samples",
        minimum=1,
    )
    _number(
        _required(section, "minimum_effective_samples_per_dimension", "te"),
        "te.minimum_effective_samples_per_dimension",
        minimum=0.0,
        inclusive_minimum=False,
    )
    _integer(
        _required(section, "autocorrelation_max_lag", "te"),
        "te.autocorrelation_max_lag",
        minimum=1,
    )
    _bool(
        _required(section, "select_parameters_on_baseline", "te"),
        "te.select_parameters_on_baseline",
    )
    selection_value = _string(
        _required(section, "selection_value", "te"), "te.selection_value"
    )
    if selection_value not in {"te_cv_bits", "te_raw_bits", "te_bits"}:
        raise ConfigError(
            "te.selection_value must be te_cv_bits, te_raw_bits, or te_bits"
        )


def _validate_cross_null(section: dict[str, Any]) -> None:
    enabled = _bool(
        _required(section, "enabled", "cross_trajectory_null"),
        "cross_trajectory_null.enabled",
    )
    repetitions = _integer(
        _required(section, "repetitions", "cross_trajectory_null"),
        "cross_trajectory_null.repetitions",
        minimum=0,
    )
    if enabled and repetitions < 1:
        raise ConfigError(
            "cross_trajectory_null.repetitions must be positive when enabled"
        )
    randomized = _bool(
        _required(section, "randomize_within_caliper", "cross_trajectory_null"),
        "cross_trajectory_null.randomize_within_caliper",
    )
    if enabled and repetitions > 1 and not randomized:
        raise ConfigError(
            "cross_trajectory_null.randomize_within_caliper must be true when "
            "requesting multiple null repetitions; deterministic nearest-neighbour "
            "matching would otherwise repeat the same pairing"
        )
    _number(
        _required(section, "length_tolerance_fraction", "cross_trajectory_null"),
        "cross_trajectory_null.length_tolerance_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if float(section["length_tolerance_fraction"]) >= 1.0:
        raise ConfigError(
            "cross_trajectory_null.length_tolerance_fraction must be < 1"
        )
    _number(
        _required(section, "match_caliper_sd", "cross_trajectory_null"),
        "cross_trajectory_null.match_caliper_sd",
        minimum=0.0,
        inclusive_minimum=False,
    )


def _validate_statistics(section: dict[str, Any]) -> None:
    unit = _string(
        _required(section, "independent_unit", "statistics"),
        "statistics.independent_unit",
    )
    if unit != "network_id":
        raise ConfigError("statistics.independent_unit must be 'network_id'")
    aggregation = _string(
        _required(section, "primary_module_aggregation", "statistics"),
        "statistics.primary_module_aggregation",
    )
    if aggregation != "mean":
        raise ConfigError("statistics.primary_module_aggregation must be 'mean'")
    _integer(
        _required(section, "bootstrap_draws", "statistics"),
        "statistics.bootstrap_draws",
        minimum=1,
    )
    _number(
        _required(section, "confidence_level", "statistics"),
        "statistics.confidence_level",
        minimum=0.0,
        maximum=1.0,
        inclusive_minimum=False,
    )
    if float(section["confidence_level"]) >= 1.0:
        raise ConfigError("statistics.confidence_level must be < 1")
    _integer(
        _required(section, "exact_sign_flip_max_networks", "statistics"),
        "statistics.exact_sign_flip_max_networks",
        minimum=1,
    )
    _integer(
        _required(section, "monte_carlo_sign_flips", "statistics"),
        "statistics.monte_carlo_sign_flips",
        minimum=1,
    )
    primary = _string(
        _required(section, "primary_multiplicity", "statistics"),
        "statistics.primary_multiplicity",
    )
    if primary != "holm":
        raise ConfigError("statistics.primary_multiplicity must be 'holm'")
    sensitivity = _string(
        _required(section, "sensitivity_multiplicity", "statistics"),
        "statistics.sensitivity_multiplicity",
    )
    if sensitivity not in {"max_t", "fdr_bh"}:
        raise ConfigError(
            "statistics.sensitivity_multiplicity must be 'max_t' or 'fdr_bh'"
        )
    targets = _required(section, "primary_targets", "statistics")
    if not isinstance(targets, list) or not targets:
        raise ConfigError("statistics.primary_targets must be a non-empty list")
    identities: list[tuple[str, str, str]] = []
    for index, raw_target in enumerate(targets):
        path = f"statistics.primary_targets[{index}]"
        target = _mapping(raw_target, path)
        stage = _string(_required(target, "stage", path), f"{path}.stage")
        environment = _string(
            _required(target, "environment", path), f"{path}.environment"
        )
        session = _required(target, "session", path)
        if isinstance(session, bool) or not isinstance(session, (str, int)):
            raise ConfigError(f"{path}.session must be a string or integer")
        identities.append((stage, environment, str(session)))
    if len(set(identities)) != len(identities):
        raise ConfigError("statistics.primary_targets contains duplicate targets")


def _validate_execution(section: dict[str, Any]) -> None:
    for key in (
        "run_primary",
        "run_sensitivity",
        "write_intermediate_scores",
        "fail_on_qc",
    ):
        _bool(_required(section, key, "execution"), f"execution.{key}")
    if not (section["run_primary"] or section["run_sensitivity"]):
        raise ConfigError(
            "At least one of execution.run_primary or execution.run_sensitivity "
            "must be true"
        )


@dataclass(frozen=True)
class Config(Mapping[str, Any]):
    """Validated analysis configuration with mapping-style access."""

    schema_version: int
    random_seed: int
    data: dict[str, Any]
    baseline: dict[str, Any]
    segmentation: dict[str, Any]
    representation: dict[str, Any]
    conditioning: dict[str, Any]
    te: dict[str, Any]
    cross_trajectory_null: dict[str, Any]
    statistics: dict[str, Any]
    execution: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        source = Path(path).resolve()
        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle, object_pairs_hook=_no_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigError("The configuration root must be a JSON object")
        return cls.from_dict(payload, source_path=source)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, source_path: str | Path | None = None
    ) -> "Config":
        raw = deepcopy(dict(payload))
        for key in ("schema_version", "random_seed", *_SECTIONS):
            if key not in raw:
                raise ConfigError(f"Missing required top-level setting {key!r}")
        _apply_defaults(raw)
        known = {"schema_version", "random_seed", *_SECTIONS}
        instance = cls(
            schema_version=raw["schema_version"],
            random_seed=raw["random_seed"],
            **{section: raw[section] for section in _SECTIONS},
            extras={key: value for key, value in raw.items() if key not in known},
            source_path=Path(source_path).resolve() if source_path is not None else None,
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        version = _integer(self.schema_version, "schema_version", minimum=1)
        if version != 1:
            raise ConfigError(f"Unsupported schema_version {version}; expected 1")
        _integer(self.random_seed, "random_seed", minimum=0)
        sections = {name: _mapping(getattr(self, name), name) for name in _SECTIONS}
        _validate_data(sections["data"])
        _validate_baseline(sections["baseline"])
        _validate_segmentation(sections["segmentation"])
        _validate_representation(sections["representation"])
        _validate_conditioning(sections["conditioning"])
        _validate_te(sections["te"])
        _validate_cross_null(sections["cross_trajectory_null"])
        _validate_statistics(sections["statistics"])
        _validate_execution(sections["execution"])

        baseline_identity = (
            str(self.baseline["stage"]),
            str(self.baseline["environment"]),
            str(self.baseline["session"]),
        )
        target_identities = {
            (str(item["stage"]), str(item["environment"]), str(item["session"]))
            for item in self.statistics["primary_targets"]
        }
        if baseline_identity in target_identities:
            raise ConfigError("statistics.primary_targets must not include the baseline")
        has_bvc_branch = "behavior_bvc_conditioned" in self.conditioning["branches"]
        if has_bvc_branch:
            if "bvc" not in self.data["array_keys"]:
                raise ConfigError(
                    "The BVC-conditioned branch requires data.array_keys.bvc"
                )
            if not bool(self.data.get("require_bvc", False)):
                raise ConfigError(
                    "The BVC-conditioned branch requires data.require_bvc=true so "
                    "missing BVC arrays fail validation"
                )
        if self.te["bias_shuffles"] > 0:
            minimum_for_shift = 2 * int(self.te["minimum_shift_frames"]) + 1
            if self.segmentation["minimum_segment_frames"] < minimum_for_shift:
                raise ConfigError(
                    "segmentation.minimum_segment_frames must be at least "
                    f"{minimum_for_shift} when circular-shift bias nulls are enabled"
                )
        if (
            self.te["select_parameters_on_baseline"]
            and self.te["selection_value"] == "te_cv_bits"
            and not self.te["compute_cv_score"]
        ):
            raise ConfigError(
                "te.compute_cv_score must be true when baseline selection uses "
                "te.selection_value='te_cv_bits'"
            )
        if self.te["effect_value"] == "te_bias_corrected_bits":
            if int(self.te["bias_shuffles"]) < 1:
                raise ConfigError(
                    "te.effect_value='te_bias_corrected_bits' requires "
                    "te.bias_shuffles > 0"
                )
            if self.execution["run_sensitivity"] and int(
                self.te["sensitivity_bias_shuffles"]
            ) < 1:
                raise ConfigError(
                    "Bias-corrected downstream effects require "
                    "te.sensitivity_bias_shuffles > 0 when sensitivity runs; "
                    "otherwise the primary and sensitivity estimands differ"
                )

        # Segment length is deliberately not cross-validated against the largest
        # lag here. Baseline selection evaluates multiple delays/histories, and
        # each candidate pool must count its own usable lagged samples.

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "random_seed": self.random_seed,
            **{name: deepcopy(getattr(self, name)) for name in _SECTIONS},
        }
        payload.update(deepcopy(self.extras))
        return payload

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a config-relative path without requiring the current directory."""

        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        base = self.source_path.parent if self.source_path is not None else Path.cwd()
        return (base / candidate).resolve()

    def __getitem__(self, key: str) -> Any:
        if key in {"schema_version", "random_seed", *_SECTIONS}:
            return getattr(self, key)
        return self.extras[key]

    def __iter__(self) -> Iterator[str]:
        yield "schema_version"
        yield "random_seed"
        yield from _SECTIONS
        yield from self.extras

    def __len__(self) -> int:
        return 2 + len(_SECTIONS) + len(self.extras)
