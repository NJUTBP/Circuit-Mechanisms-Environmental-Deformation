"""Regenerate manuscript and supplementary figures from the packaged data."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGURES = [f"main_figure_{n}" for n in range(1, 7)]
FIGURES += [f"supp_figure_{n}" for n in range(1, 17)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=FIGURES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Reproduced")
    args = parser.parse_args()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for index, name in enumerate(args.figures, start=1):
        target = args.output_dir.resolve() / ("Main_figs" if name.startswith("main") else "Supp_figs")
        print(f"[{index}/{len(args.figures)}] {name}", flush=True)
        subprocess.run([sys.executable, str(HERE / f"plot_{name}.py"),
                        "--output-dir", str(target)], check=True, env=environment)


if __name__ == "__main__":
    main()
