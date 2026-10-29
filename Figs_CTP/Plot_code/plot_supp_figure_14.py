"""Coupling-strength response heatmaps."""
import argparse
from pathlib import Path
from strength_common import plot_strength

def plot(output_dir=None):
    return plot_strength(14,output_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    print(plot(parser.parse_args().output_dir))
