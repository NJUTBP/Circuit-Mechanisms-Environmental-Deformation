"""Convenience entry point for the selected-network data_simu TE interface."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from te_population_gcmi.data_simu_interface import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
