"""Small checks for the reviewer model and first-batch initialization.

These checks do not execute the long network simulation.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from review_pipeline.config import ConfigError, PipelineConfig
from review_pipeline.model import NetworkState, initial_grid_activity, initialize_network
from review_pipeline.plan import build_seed_records, build_tasks
from review_pipeline.simulation import grid_warmup_velocities


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.model.json"
FIRST_BATCH_DATA = (
    Path(__file__).resolve().parents[3]
    / "data_simu" / "reviewer_independent_networks_gc_shared_init_psum18_v4"
)


class ReviewerGridInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["model"]

    def test_seeded_three_point_activity(self) -> None:
        grids = initial_grid_activity(self.model, seed=1234)
        replay = initial_grid_activity(self.model, seed=1234)
        changed = initial_grid_activity(self.model, seed=1235)
        selected = np.zeros((64, 64), dtype=bool)
        selected[[32, 32, 32], [15, 32, 49]] = True
        for grid, same in zip(grids, replay):
            np.testing.assert_array_equal(grid, same)
            np.testing.assert_array_equal(grid[selected], np.ones(3))
            self.assertEqual(np.count_nonzero(grid == 1.0), 3)
            self.assertTrue(np.all((grid[~selected] >= 0.0) & (grid[~selected] < 0.01)))
        self.assertFalse(np.array_equal(grids[0], grids[1]))
        self.assertFalse(np.array_equal(grids[0], changed[0]))

    def test_previous_activation_modes_are_rejected(self) -> None:
        for mode in ("legacy_pattern", "gc_module_v_plas_init", "seeded_noise"):
            with self.subTest(mode=mode):
                model = dict(self.model, grid_initialization=mode)
                with self.assertRaisesRegex(ValueError, "Only model.grid_initialization"):
                    initial_grid_activity(model, seed=1234)
                config = PipelineConfig(CONFIG_PATH)
                config.data["model"]["grid_initialization"] = mode
                with self.assertRaisesRegex(ConfigError, "three_point_seeded_background"):
                    config.validate()

    def test_activation_size_and_background_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires model.n_grid=64"):
            initial_grid_activity(dict(self.model, n_grid=32), seed=1234)
        for maximum in (0.0, -0.01, 0.02, float("nan"), float("inf")):
            with self.subTest(maximum=maximum):
                with self.assertRaisesRegex(ValueError, "initial_grid_background_max"):
                    initial_grid_activity(
                        dict(self.model, initial_grid_background_max=maximum), seed=1234
                    )

    def test_seeded_small_warmup_velocity(self) -> None:
        velocities = grid_warmup_velocities(self.model, seed=1234)
        np.testing.assert_array_equal(
            velocities, grid_warmup_velocities(self.model, seed=1234)
        )
        self.assertFalse(
            np.array_equal(velocities, grid_warmup_velocities(self.model, seed=1235))
        )
        self.assertEqual(velocities.shape, (3, 2))
        self.assertTrue(np.all(np.abs(velocities) < 0.01))

    def test_requested_rotation_values_are_recorded(self) -> None:
        self.assertEqual(
            self.model["grid_velocity_rotations_deg"], [-15.0, 15.0, -10.0]
        )
        self.assertEqual(
            self.model["grid_orientations_deg"], [15.0, -15.0, -10.0]
        )

    def test_each_network_records_activity_and_warmup_seed(self) -> None:
        config = PipelineConfig(CONFIG_PATH)
        self.assertEqual(config.n_networks, 8)
        relevant = [
            record for record in build_seed_records(config)
            if record.purpose in {
                "initial_grid_activity", "initial_grid_warmup_velocity"
            }
        ]
        self.assertEqual(len(relevant), 2 * config.data["n_networks"])
        for network_id in config.network_ids:
            pair = [record for record in relevant if record.network_id == network_id]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0].seed, pair[1].seed)

    def test_eight_network_task_plan(self) -> None:
        config = PipelineConfig(CONFIG_PATH)
        self.assertEqual(config.network_ids, list(range(8)))
        tasks = build_tasks(config)
        self.assertEqual(
            {stage: len(items) for stage, items in tasks.items()},
            {"init": 8, "baseline": 8, "initial_exposure": 48, "familiarization": 16},
        )
        for stage_tasks in tasks.values():
            self.assertEqual(
                {task["network_id"] for task in stage_tasks}, set(range(8))
            )

    def test_first_batch_sampler_keeps_gamma_rng_sequence(self) -> None:
        seed = 3987358787
        model = dict(self.model, n_grid=8)
        rng = np.random.default_rng(seed)
        n_boundary = int(model["n_boundary"])
        n_place = int(model["n_place"])
        inputs = int(model["bvc_inputs_per_place"])
        stubs = np.repeat(np.arange(n_boundary), n_place * inputs // n_boundary)
        for _ in range(1000):
            selected = rng.permutation(stubs).reshape(n_place, inputs)
            ordered = np.sort(selected, axis=1)
            if np.all(np.diff(ordered, axis=1) != 0):
                break
        else:
            self.fail("First-batch reference seed did not produce a simple graph")
        expected_bvc = np.zeros((n_boundary, n_place))
        expected_bvc[selected, np.arange(n_place)[:, None]] = 1.0
        actual = initialize_network(model, seed)
        np.testing.assert_array_equal(actual.w_bvc_to_place, expected_bvc)
        np.testing.assert_array_equal(expected_bvc.sum(axis=0), np.full(n_place, inputs))
        np.testing.assert_array_equal(
            expected_bvc.sum(axis=1), np.full(n_boundary, n_place * inputs // n_boundary)
        )

        gamma = model["initial_weight_gamma"]
        multipliers = model["initial_weight_multipliers"]
        expected_pc = float(multipliers["place_recurrent"]) * rng.gamma(
            shape=gamma["shape"], scale=gamma["scale"], size=(n_place, n_place)
        )
        np.fill_diagonal(expected_pc, 0.0)
        np.testing.assert_array_equal(actual.w_place_recurrent, expected_pc)
        for name, matrices in (
            ("grid_to_place", actual.w_grid_to_place),
            ("place_to_grid", actual.w_place_to_grid),
        ):
            for matrix in matrices:
                expected = float(multipliers[name]) * rng.gamma(
                    shape=gamma["shape"], scale=gamma["scale"],
                    size=(int(model["n_grid"]) ** 2, n_place),
                )
                np.testing.assert_array_equal(matrix, expected)

    def test_saved_first_batch_initial_weights(self) -> None:
        paths = [FIRST_BATCH_DATA / "networks" / f"network_{i:03d}" / "init"
                 for i in range(8)]
        if not all((path / "initial_weights.npz").is_file()
                   and (path / "initial_weights.json").is_file() for path in paths):
            self.skipTest("Optional saved first-batch data are not present")
        for network_id, path in enumerate(paths):
            with self.subTest(network_id=network_id):
                metadata = json.loads(
                    (path / "initial_weights.json").read_text(encoding="utf-8")
                )
                actual = initialize_network(self.model, seed=int(metadata["seed"]))
                saved = NetworkState.load(path / "initial_weights.npz")
                np.testing.assert_array_equal(actual.w_bvc_to_place, saved.w_bvc_to_place)
                for regenerated, historical in zip(
                    (actual.w_place_recurrent, *actual.w_grid_to_place,
                     *actual.w_place_to_grid),
                    (saved.w_place_recurrent, *saved.w_grid_to_place,
                     *saved.w_place_to_grid),
                ):
                    np.testing.assert_allclose(
                        regenerated, historical, rtol=1e-14, atol=0.0
                    )


if __name__ == "__main__":
    unittest.main()
