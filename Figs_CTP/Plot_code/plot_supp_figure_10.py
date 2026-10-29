"""Representative place-cell rate maps."""
import argparse
from pathlib import Path
from rate_map_common import plot_place

def plot(output_dir=None):
    return plot_place(10,output_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    print(plot(parser.parse_args().output_dir))
