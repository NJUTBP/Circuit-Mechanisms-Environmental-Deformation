from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np


class ConfigError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PipelineConfig:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()
        with self.path.open("r", encoding="utf-8") as handle:
            self.data = json.load(handle)
        self._apply_environment_overrides()
        self.validate()

    def _apply_environment_overrides(self) -> None:
        data_root = os.environ.get("RESCALE_DATA_ROOT")
        output_root = os.environ.get("RESCALE_OUTPUT_ROOT")
        if data_root:
            self.data["data_root"] = data_root
        if output_root:
            self.data["output_root"] = output_root

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.data).encode("utf-8")).hexdigest()

    @cached_property
    def pipeline_hash(self) -> str:
        """Fingerprint executable pipeline code and the selected local analyses."""
        digest = hashlib.sha256()
        package_root = Path(__file__).resolve().parent
        files = list(package_root.glob("*.py"))
        runner = package_root.parent / "run_pipeline.py"
        if runner.exists():
            files.append(runner)
        legacy_root = self.resolve_path(
            self.data["analysis"]["legacy_function_dir"], base=self.project_root
        )
        if legacy_root.exists():
            files.extend(legacy_root.glob("*.py"))
        for source in sorted(files, key=lambda value: str(value).lower()):
            digest.update(source.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @cached_property
    def runtime_details(self) -> dict[str, Any]:
        distributions = [
            "numpy", "scipy", "pandas", "scikit-learn", "umap-learn",
            "ratinabox", "opexebo",
        ]
        versions: dict[str, str] = {}
        for distribution in distributions:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = "not-installed"
        return {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
            "packages": versions,
        }

    @cached_property
    def runtime_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.runtime_details).encode("utf-8")
        ).hexdigest()

    @property
    def missing_dependencies(self) -> list[str]:
        required = {"numpy", "scipy", "pandas", "ratinabox", "opexebo"}
        if "umap" in self.data["analysis"]["te"].get("embeddings", []):
            required.add("umap-learn")
        versions = self.runtime_details["packages"]
        return sorted(
            distribution
            for distribution in required
            if versions.get(distribution) == "not-installed"
        )

    @cached_property
    def input_details(self) -> dict[str, Any]:
        keys: list[str] = []
        if self.data["model"]["bvc_tuning"]["source"] == "fixed_files":
            keys.extend(["bvc_distance", "bvc_angle"])
            if "bvc_sigma_angle" in self.data["inputs"]:
                keys.append("bvc_sigma_angle")
        if self.data["analysis"]["grid_sample"]["mode"] == "legacy_file":
            keys.append("grid_indices")
        details: dict[str, Any] = {}
        for key in keys:
            path = self.input_path(key)
            details[key] = (
                {"bytes": path.stat().st_size, "sha256": _sha256_path(path)}
                if path.exists()
                else {"missing": True}
            )
        return details

    @cached_property
    def input_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.input_details).encode("utf-8")
        ).hexdigest()

    @property
    def run_hash(self) -> str:
        message = (
            f"{self.config_hash}\0{self.pipeline_hash}\0{self.runtime_hash}"
            f"\0{self.input_hash}"
        ).encode("ascii")
        return hashlib.sha256(message).hexdigest()

    @property
    def hash(self) -> str:
        """Backward-compatible alias used internally for the complete run fingerprint."""
        return self.run_hash

    def accepted_run_hashes(self, stage: str) -> set[str]:
        """Return current plus explicitly verified upstream hashes for a stage."""

        accepted = {self.run_hash}
        path = self.output_root / "upstream_compatibility.json"
        if not path.is_file():
            return accepted
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return accepted
        if (
            payload.get("status") == "verified"
            and payload.get("target_run_sha256") == self.run_hash
            and stage in payload.get("stages", [])
        ):
            source_hash = payload.get("source_run_sha256")
            if isinstance(source_hash, str) and len(source_hash) == 64:
                accepted.add(source_hash)
            stage_sources = payload.get("source_run_sha256_by_stage", {})
            if isinstance(stage_sources, dict):
                hashes = stage_sources.get(stage, [])
                if isinstance(hashes, list):
                    accepted.update(
                        value for value in hashes
                        if isinstance(value, str) and len(value) == 64
                    )
        return accepted

    @property
    def master_seed(self) -> int:
        return int(self.data["master_seed"])

    @property
    def n_networks(self) -> int:
        return int(self.data["n_networks"])

    @property
    def network_ids(self) -> list[int]:
        return list(range(self.n_networks))

    @property
    def trajectory_ids(self) -> list[int]:
        return [int(value) for value in self.data["evaluation"]["trajectory_ids"]]

    @property
    def familiarization_trajectory_ids(self) -> list[int]:
        """Return the Session-0 trajectory IDs continued during familiarization.

        Older configurations used every evaluation trajectory as an independent
        branch. New single-trajectory configurations select one explicitly while
        baseline and initial-exposure replication remain unchanged.
        """

        values = self.data["familiarization"].get("trajectory_ids")
        if values is None:
            return self.trajectory_ids
        return [int(value) for value in values]

    @property
    def config_dir(self) -> Path:
        return self.path.parent

    @property
    def project_root(self) -> Path:
        value = self.data.get("project_root", "..")
        return self.resolve_path(value, base=self.config_dir)

    @property
    def data_root(self) -> Path:
        return self.resolve_path(self.data["data_root"], base=self.config_dir)

    @property
    def output_root(self) -> Path:
        return self.resolve_path(self.data["output_root"], base=self.config_dir)

    def resolve_path(self, value: str | os.PathLike[str], base: Path | None = None) -> Path:
        path = Path(os.path.expandvars(str(value))).expanduser()
        if not path.is_absolute():
            path = (base or self.config_dir) / path
        return path.resolve()

    def input_path(self, key: str) -> Path:
        return self.resolve_path(self.data["inputs"][key], base=self.data_root)

    def environment(self, env_id: str) -> dict[str, Any]:
        for env in self.data["environments"]:
            if env["id"] == env_id:
                return env
        raise ConfigError(f"Unknown environment: {env_id}")

    @property
    def baseline_environment(self) -> dict[str, Any]:
        matches = [env for env in self.data["environments"] if env.get("role") == "baseline"]
        if len(matches) != 1:
            raise ConfigError("Exactly one environment must have role='baseline'")
        return matches[0]

    def stage_environments(self, stage: str) -> list[str]:
        if stage == "initial_exposure":
            return list(self.data["initial_exposure"]["environments"])
        if stage == "familiarization":
            return list(self.data["familiarization"]["environments"])
        raise ConfigError(f"Stage has no environment list: {stage}")

    def validate(self) -> None:
        required = [
            "schema_version", "experiment_name", "master_seed", "n_networks",
            "data_root", "output_root", "inputs", "model", "environments",
            "initialization", "evaluation", "initial_exposure", "familiarization",
            "analysis", "storage", "slurm",
        ]
        missing = [key for key in required if key not in self.data]
        if missing:
            raise ConfigError(f"Missing configuration keys: {missing}")
        if int(self.data["schema_version"]) != 1:
            raise ConfigError("Only schema_version=1 is supported")
        if not 1 <= self.n_networks <= 10000:
            raise ConfigError("n_networks must be between 1 and 10000")
        if len(set(self.trajectory_ids)) < 1:
            raise ConfigError("At least one evaluation trajectory ID is required")
        if len(self.trajectory_ids) != len(set(self.trajectory_ids)):
            raise ConfigError("evaluation.trajectory_ids must be unique")
        env_ids = [env["id"] for env in self.data["environments"]]
        if len(env_ids) != len(set(env_ids)):
            raise ConfigError("Environment IDs must be unique")
        self.baseline_environment
        for env in self.data["environments"]:
            if float(env["width_m"]) <= 0 or float(env["height_m"]) <= 0:
                raise ConfigError(f"Environment dimensions must be positive: {env['id']}")
        for stage in ("initial_exposure", "familiarization"):
            values = self.stage_environments(stage)
            if len(values) != len(set(values)):
                raise ConfigError(f"Duplicate {stage} environments are not allowed")
            unknown = set(self.stage_environments(stage)) - set(env_ids)
            if unknown:
                raise ConfigError(f"Unknown {stage} environments: {sorted(unknown)}")
        if self.baseline_environment["id"] in self.stage_environments("initial_exposure"):
            raise ConfigError("The baseline environment must not be duplicated in initial_exposure")
        missing_exposure = (
            set(self.stage_environments("familiarization"))
            - set(self.stage_environments("initial_exposure"))
        )
        if missing_exposure:
            raise ConfigError(
                "Every familiarization environment needs a Session-0 initial exposure: "
                f"{sorted(missing_exposure)}"
            )
        model = self.data["model"]
        required_model_keys = [
            "n_boundary", "n_place", "n_grid", "bvc_inputs_per_place",
            "dt_s", "duration_s", "record_every_steps",
            "plasticity_every_steps", "warmup_steps", "warmup_velocity",
            "tau_grid_s", "tau_place_s", "grid_kernel_shift",
            "grid_kernel_wavelength", "grid_kernel_gamma_ratio",
            "grid_kernel_centre_gain", "grid_velocity_gains",
            "grid_orientations_deg", "grid_initialization", "bvc_tuning",
            "bvc_drive_gain", "grid_to_place_gain", "place_threshold",
            "place_inhibition_mode", "place_global_inhibition",
            "target_place_sparsity", "place_activity_floor",
            "initial_weight_gamma", "initial_weight_multipliers",
            "learning_rates", "normalization",
        ]
        missing_model_keys = [key for key in required_model_keys if key not in model]
        if missing_model_keys:
            raise ConfigError(f"Missing model configuration keys: {missing_model_keys}")
        n_boundary = int(model["n_boundary"])
        n_place = int(model["n_place"])
        inputs_per_place = int(model["bvc_inputs_per_place"])
        if min(n_boundary, n_place, inputs_per_place) < 1:
            raise ConfigError("n_boundary, n_place, and bvc_inputs_per_place must be positive")
        if inputs_per_place > n_boundary:
            raise ConfigError("bvc_inputs_per_place cannot exceed n_boundary")
        if (n_place * inputs_per_place) % n_boundary:
            raise ConfigError(
                "n_place * bvc_inputs_per_place must be divisible by n_boundary"
            )
        if int(model["n_grid"]) % 2:
            raise ConfigError("n_grid must be even")
        for key in ("grid_velocity_gains", "grid_orientations_deg"):
            try:
                values = [float(value) for value in model[key]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(f"model.{key} must contain three numbers") from exc
            if len(values) != 3:
                raise ConfigError(f"model.{key} must contain exactly three values")
        if "grid_velocity_rotations_deg" in model:
            try:
                rotations = np.asarray(
                    model["grid_velocity_rotations_deg"], dtype=float
                )
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "model.grid_velocity_rotations_deg must contain three numbers"
                ) from exc
            if rotations.shape != (3,) or not np.all(np.isfinite(rotations)):
                raise ConfigError(
                    "model.grid_velocity_rotations_deg must contain three finite values"
                )
        if model.get("grid_initialization") != "three_point_seeded_background":
            raise ConfigError(
                "model.grid_initialization must be three_point_seeded_background "
                "in the reviewer code"
            )
        maximum = float(model.get("initial_grid_background_max", float("nan")))
        if not np.isfinite(maximum) or not 0.0 < maximum <= 0.01:
            raise ConfigError(
                "model.initial_grid_background_max must be in (0, 0.01]"
            )
        if int(model["n_grid"]) != 64:
            raise ConfigError(
                f"model.grid_initialization={model['grid_initialization']!r} "
                "requires n_grid=64"
            )
        if "grid_warmup_velocities" in model:
            try:
                warmup_velocities = np.asarray(
                    model["grid_warmup_velocities"], dtype=float
                )
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "model.grid_warmup_velocities must contain three [vx, vy] pairs"
                ) from exc
            if warmup_velocities.shape != (3, 2):
                raise ConfigError(
                    "model.grid_warmup_velocities must have shape [3, 2]"
                )
            if not np.all(np.isfinite(warmup_velocities)):
                raise ConfigError("model.grid_warmup_velocities must be finite")
        warmup_mode = model.get("grid_warmup_velocity_mode", "fixed")
        if warmup_mode not in {"fixed", "seeded_small"}:
            raise ConfigError(
                "model.grid_warmup_velocity_mode must be fixed or seeded_small"
            )
        if warmup_mode == "seeded_small":
            maximum = float(model.get("grid_warmup_velocity_max", float("nan")))
            if not np.isfinite(maximum) or not 0.0 < maximum <= 0.01:
                raise ConfigError(
                    "model.grid_warmup_velocity_max must be in (0, 0.01]"
                )
        if int(model["record_every_steps"]) < 1:
            raise ConfigError("record_every_steps must be positive")
        if float(model["dt_s"]) <= 0 or float(model["duration_s"]) <= 0:
            raise ConfigError("dt_s and duration_s must be positive")
        tuning = model.get("bvc_tuning")
        if not isinstance(tuning, dict):
            raise ConfigError("model.bvc_tuning must be configured")
        tuning_source = tuning.get("source")
        if tuning_source not in {"generated_per_network", "fixed_files"}:
            raise ConfigError(
                "model.bvc_tuning.source must be generated_per_network or fixed_files"
            )
        if float(tuning.get("distance_scale", 1.0)) <= 0:
            raise ConfigError("model.bvc_tuning.distance_scale must be positive")
        if tuning_source == "generated_per_network":
            distance = tuning.get("distance", {})
            angle = tuning.get("angle", {})
            if distance.get("distribution") != "truncnorm":
                raise ConfigError(
                    "Generated BVC distance distribution must be truncnorm"
                )
            if angle.get("distribution") != "uniform":
                raise ConfigError("Generated BVC angle distribution must be uniform")
            try:
                a = float(distance["a"])
                b = float(distance["b"])
                loc_m = float(distance["loc_m"])
                scale_m = float(distance["scale_m"])
                low_deg = float(angle.get("low_deg", 0.0))
                high_deg = float(angle.get("high_deg", 360.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError("Invalid generated BVC tuning parameters") from exc
            if not a < b or scale_m <= 0:
                raise ConfigError("BVC truncnorm requires a < b and scale_m > 0")
            if loc_m + a * scale_m < 0:
                raise ConfigError("Generated BVC distances cannot include negative values")
            if not 0.0 <= low_deg < high_deg <= 360.0:
                raise ConfigError(
                    "Generated BVC angle bounds must satisfy 0 <= low < high <= 360"
                )
            angular_width = tuning.get("angular_width", {})
            radial_width = tuning.get("radial_width", {})
            if angular_width.get("distribution") != "uniform":
                raise ConfigError("Generated BVC angular widths must be uniform")
            if radial_width.get("distribution") != "diverging":
                raise ConfigError("BVC radial widths must use the diverging rule")
            try:
                width_low = float(angular_width["low_deg"])
                width_high = float(angular_width["high_deg"])
                xi_m = float(radial_width["xi_m"])
                beta = float(radial_width["beta"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError("Invalid BVC width parameters") from exc
            if not 0.0 < width_low < width_high <= 180.0:
                raise ConfigError("BVC angular widths must satisfy 0 < low < high <= 180")
            if xi_m <= 0.0 or beta <= 0.0:
                raise ConfigError("BVC radial xi and beta must be positive")
        gamma_cfg = model.get("initial_weight_gamma", {})
        multipliers = model.get("initial_weight_multipliers", {})
        try:
            gamma_shape = float(gamma_cfg["shape"])
            gamma_scale = float(gamma_cfg["scale"])
            initial_multipliers = [
                float(multipliers[key])
                for key in ("place_recurrent", "grid_to_place", "place_to_grid")
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("Invalid initial Gamma-weight configuration") from exc
        if gamma_shape <= 0 or gamma_scale <= 0 or min(initial_multipliers) <= 0:
            raise ConfigError("Initial Gamma parameters and multipliers must be positive")
        norms = model.get("normalization", {})
        required_norms = (
            "bvc_to_place_incoming", "grid_to_place_incoming",
            "place_to_grid_incoming", "place_recurrent_incoming",
        )
        try:
            normalization_targets = [float(norms[key]) for key in required_norms]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("Invalid connection-weight normalization targets") from exc
        if min(normalization_targets) <= 0:
            raise ConfigError("Connection-weight normalization targets must be positive")
        for key in ("place_to_grid_max_outgoing", "place_recurrent_max_single"):
            value = norms.get(key)
            if value is not None and float(value) <= 0:
                raise ConfigError(f"model.normalization.{key} must be positive or null")
        pathways = (
            "bvc_to_place", "grid_to_place", "place_to_grid", "place_recurrent"
        )
        for section_name, source in (
            ("model.learning_rates", model.get("learning_rates", {})),
            (
                "initialization.plasticity",
                self.data["initialization"].get("plasticity", {}),
            ),
        ):
            missing_pathways = [key for key in pathways if key not in source]
            if missing_pathways:
                raise ConfigError(
                    f"{section_name} is missing pathways: {missing_pathways}"
                )
            try:
                values = [float(source[key]) for key in pathways]
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{section_name} values must be numeric") from exc
            if min(values) < 0:
                raise ConfigError(f"{section_name} values cannot be negative")
        analysis = self.data["analysis"]
        if float(analysis["rate_map_bin_m"]) <= 0:
            raise ConfigError("analysis.rate_map_bin_m must be positive")
        if float(analysis["rate_map_sigma_bins"]) <= 0:
            raise ConfigError("analysis.rate_map_sigma_bins must be positive")
        if float(analysis["place_field_rate_map_sigma_bins"]) <= 0:
            raise ConfigError(
                "analysis.place_field_rate_map_sigma_bins must be positive"
            )
        if float(analysis["rescale_scale_range_cm"]) < 0:
            raise ConfigError("analysis.rescale_scale_range_cm cannot be negative")
        if float(analysis["rescale_scale_step_cm"]) <= 0:
            raise ConfigError("analysis.rescale_scale_step_cm must be positive")
        if int(analysis.get("correlation_min_overlap_bins", 20)) < 2:
            raise ConfigError("correlation_min_overlap_bins must be at least 2")
        if not 0.0 <= float(
            analysis.get("spatial_information_min_occupancy_fraction", 0.0)
        ) < 1.0:
            raise ConfigError(
                "spatial_information_min_occupancy_fraction must lie in [0, 1)"
            )
        raw_policy = self.data["storage"]["raw_activity"]
        if not isinstance(
            self.data["initial_exposure"].get("retain_raw_activity", False), bool
        ):
            raise ConfigError("initial_exposure.retain_raw_activity must be a boolean")
        if raw_policy not in {"none", "qa", "all", "selective"}:
            raise ConfigError(
                "storage.raw_activity must be none, qa, all, or selective"
            )
        if raw_policy == "selective":
            selection = self.data["storage"].get("raw_activity_selection")
            if not isinstance(selection, dict):
                raise ConfigError(
                    "storage.raw_activity_selection is required for selective storage"
                )
            selected_networks = selection.get("network_ids", "all")
            if selected_networks != "all":
                try:
                    selected_networks = [int(value) for value in selected_networks]
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        "raw_activity_selection.network_ids must be 'all' or a list"
                    ) from exc
                invalid_networks = [
                    value for value in selected_networks if value not in self.network_ids
                ]
                if invalid_networks:
                    raise ConfigError(
                        "raw activity network IDs outside configured networks: "
                        f"{invalid_networks}"
                    )
            stage_rules = selection.get("stages")
            if not isinstance(stage_rules, dict) or not stage_rules:
                raise ConfigError(
                    "raw_activity_selection.stages must contain at least one stage"
                )
            allowed_raw_stages = {"baseline", "initial_exposure", "familiarization"}
            unknown_raw_stages = set(stage_rules) - allowed_raw_stages
            if unknown_raw_stages:
                raise ConfigError(
                    f"Unknown raw activity stages: {sorted(unknown_raw_stages)}"
                )
            for raw_stage, rule in stage_rules.items():
                if not isinstance(rule, dict):
                    raise ConfigError(f"Raw activity rule for {raw_stage} must be an object")
                selected_envs = rule.get("environments", "all")
                if selected_envs != "all":
                    unknown_raw_envs = set(selected_envs) - set(env_ids)
                    if unknown_raw_envs:
                        raise ConfigError(
                            f"Unknown raw activity environments: {sorted(unknown_raw_envs)}"
                        )
                selected_sessions = rule.get("sessions", "all")
                if selected_sessions != "all":
                    try:
                        selected_sessions = [int(value) for value in selected_sessions]
                    except (TypeError, ValueError) as exc:
                        raise ConfigError(
                            f"Raw activity sessions for {raw_stage} must be 'all' or a list"
                        ) from exc
                    maximum = (
                        int(self.data["familiarization"]["sessions"])
                        if raw_stage == "familiarization"
                        else 0
                    )
                    invalid_raw_sessions = [
                        value for value in selected_sessions
                        if value < 0 or value > maximum
                    ]
                    if invalid_raw_sessions:
                        raise ConfigError(
                            f"Raw activity sessions outside {raw_stage} range: "
                            f"{invalid_raw_sessions}"
                        )
        trajectory_mode = self.data["familiarization"]["trajectory_mode"]
        if trajectory_mode not in {"branches", "single"}:
            raise ConfigError(
                "familiarization.trajectory_mode must be 'branches' or 'single'"
            )
        familiarization_trajectory_ids = self.familiarization_trajectory_ids
        if not familiarization_trajectory_ids:
            raise ConfigError(
                "familiarization.trajectory_ids must contain at least one ID"
            )
        if len(familiarization_trajectory_ids) != len(
            set(familiarization_trajectory_ids)
        ):
            raise ConfigError("familiarization.trajectory_ids must be unique")
        unknown_familiarization_trajectory_ids = (
            set(familiarization_trajectory_ids) - set(self.trajectory_ids)
        )
        if unknown_familiarization_trajectory_ids:
            raise ConfigError(
                "Familiarization trajectory IDs must identify completed Session-0 "
                "initial-exposure trajectories: "
                f"{sorted(unknown_familiarization_trajectory_ids)}"
            )
        if trajectory_mode == "single" and len(familiarization_trajectory_ids) != 1:
            raise ConfigError(
                "familiarization.trajectory_mode='single' requires exactly one "
                "familiarization.trajectory_ids entry"
            )
        if trajectory_mode == "branches" and len(familiarization_trajectory_ids) < 2:
            raise ConfigError(
                "familiarization.trajectory_mode='branches' requires at least two "
                "familiarization trajectory IDs"
            )
        if (
            bool(
                self.data["familiarization"].get(
                    "reuse_initial_exposure_trajectories", True
                )
            )
            and not bool(
                self.data["familiarization"].get(
                    "reuse_trajectories_across_sessions", True
                )
            )
        ):
            raise ConfigError(
                "reuse_initial_exposure_trajectories=true requires "
                "reuse_trajectories_across_sessions=true"
            )
        if bool(
            self.data["familiarization"].get(
                "reuse_initial_exposure_trajectories", True
            )
        ):
            missing_stage_zero = set(self.stage_environments("familiarization")) - set(
                self.stage_environments("initial_exposure")
            )
            if missing_stage_zero:
                raise ConfigError(
                    "Familiarization environments that reuse initial-exposure trajectories "
                    f"must also be in initial_exposure: {sorted(missing_stage_zero)}"
                )
        variants = self.data["familiarization"]["plasticity_variants"]
        variant_ids = [str(item["id"]) for item in variants]
        if not variant_ids or len(variant_ids) != len(set(variant_ids)):
            raise ConfigError("Plasticity variant IDs must be non-empty and unique")
        for variant in variants:
            multipliers = variant.get("multipliers", {})
            missing_pathways = [key for key in pathways if key not in multipliers]
            if missing_pathways:
                raise ConfigError(
                    f"Familiarization variant {variant['id']} is missing pathways: "
                    f"{missing_pathways}"
                )
            try:
                values = [float(multipliers[key]) for key in pathways]
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Familiarization variant {variant['id']} must be numeric"
                ) from exc
            if min(values) < 0:
                raise ConfigError(
                    f"Familiarization variant {variant['id']} cannot be negative"
                )
        te_cfg = self.data["analysis"]["te"]
        if bool(te_cfg["enabled"]):
            te_envs = set(te_cfg.get("environments", []))
            unknown_te_envs = te_envs - set(env_ids)
            if unknown_te_envs:
                raise ConfigError(f"Unknown TE environments: {sorted(unknown_te_envs)}")
            if self.baseline_environment["id"] not in te_envs:
                raise ConfigError(
                    "The baseline environment must be included in analysis.te.environments "
                    "so PCA/UMAP bases are fitted without leakage"
                )
            fam_session_values = [int(value) for value in te_cfg.get("familiarization_sessions", [])]
            invalid_sessions = [
                value for value in fam_session_values
                if value < 1 or value > int(self.data["familiarization"]["sessions"])
            ]
            if invalid_sessions:
                raise ConfigError(
                    f"TE familiarization sessions outside the configured range: {invalid_sessions}"
                )
            for key in ("lag_frames", "source_history", "target_history"):
                values = [int(value) for value in te_cfg[key]]
                if not values or any(value < 1 for value in values):
                    raise ConfigError(f"analysis.te.{key} must contain positive integers")
            if int(te_cfg.get("n_jobs", 1)) < 1:
                raise ConfigError("analysis.te.n_jobs must be positive")
        sample_count = int(self.data["analysis"]["grid_sample"]["n_cells"])
        grid_count = int(model["n_grid"]) ** 2
        if sample_count < 1 or sample_count > grid_count:
            raise ConfigError(
                f"analysis.grid_sample.n_cells must be between 1 and {grid_count}"
            )
        for key, value in self.data["slurm"].items():
            if key.startswith("max_parallel_") and int(value) < 1:
                raise ConfigError(f"slurm.{key} must be positive")
        upstream_reuse = self.data.get("upstream_reuse", {})
        if bool(upstream_reuse.get("enabled", False)):
            stages = upstream_reuse.get("stages")
            if not isinstance(stages, list) or not stages:
                raise ConfigError(
                    "upstream_reuse.stages must be a non-empty list when enabled"
                )
            unknown_reuse_stages = set(stages) - {"init", "baseline"}
            if unknown_reuse_stages:
                raise ConfigError(
                    "Only completed init and baseline may be reused across this "
                    f"downstream-only revision: {sorted(unknown_reuse_stages)}"
                )
        required_inputs: list[str] = []
        if tuning_source == "fixed_files":
            required_inputs.extend(["bvc_distance", "bvc_angle"])
        if self.data["analysis"]["grid_sample"]["mode"] == "legacy_file":
            required_inputs.append("grid_indices")
        missing_input_keys = [key for key in required_inputs if key not in self.data["inputs"]]
        if missing_input_keys:
            raise ConfigError(f"Missing input configuration keys: {missing_input_keys}")
        missing_inputs = [
            f"{key}={self.input_path(key)}"
            for key in required_inputs
            if not self.input_path(key).is_file()
        ]
        if missing_inputs:
            raise ConfigError(f"Missing required input files: {missing_inputs}")
        if tuning_source == "fixed_files":
            try:
                bvc_distance = np.load(self.input_path("bvc_distance"), allow_pickle=False)
                bvc_angle = np.load(self.input_path("bvc_angle"), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ConfigError(f"Could not read BVC input arrays: {exc}") from exc
            if bvc_distance.size != n_boundary or bvc_angle.size != n_boundary:
                raise ConfigError(
                    "BVC distance/angle arrays must each contain model.n_boundary values"
                )
        legacy_directory = self.resolve_path(
            self.data["analysis"]["legacy_function_dir"], base=self.project_root
        )
        required_analysis = [
            "Field_length.py", "phase_shift_grid.py", "phase_shift_place.py",
            "grid_rescale_cal.py",
        ]
        missing_analysis = [
            str(legacy_directory / filename)
            for filename in required_analysis
            if not (legacy_directory / filename).is_file()
        ]
        if missing_analysis:
            raise ConfigError(f"Missing required analysis functions: {missing_analysis}")

    def snapshot(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.data)
        payload["_resolved"] = {
            "config_path": str(self.path),
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "output_root": str(self.output_root),
            "config_sha256": self.config_hash,
            "pipeline_sha256": self.pipeline_hash,
            "runtime_sha256": self.runtime_hash,
            "runtime": self.runtime_details,
            "input_sha256": self.input_hash,
            "inputs": self.input_details,
            "run_sha256": self.run_hash,
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
