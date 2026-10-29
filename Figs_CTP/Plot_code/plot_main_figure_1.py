"""Export the supplied final diagram PDF and its 600-dpi raster files."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
from PIL import Image
from archive_common import DATA,ROOT

def poppler():
    found=os.environ.get("PDFTOPPM") or shutil.which("pdftoppm")
    if found:return str(found)
    candidates=list((Path.home()/".cache/codex-runtimes").glob("**/pdftoppm.exe"))
    if candidates:return str(candidates[0])
    raise RuntimeError("Install Poppler or set PDFTOPPM to the pdftoppm executable.")

def plot(output_dir=None):
    folder=Path(output_dir) if output_dir else ROOT/"Reproduced/Main_figs"
    folder.mkdir(parents=True,exist_ok=True)
    stem=folder/"main_figure_1"
    shutil.copy2(DATA/"main_figure_1/diagram.pdf",stem.with_suffix(".pdf"))
    subprocess.run([poppler(),"-r","600","-png","-singlefile",str(stem.with_suffix(".pdf")),str(stem)],check=True,capture_output=True)
    with Image.open(stem.with_suffix(".png")) as image:
        image.convert("RGB").save(stem.with_suffix(".tiff"),dpi=(600,600),compression="tiff_lzw")
    return stem

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    print(plot(parser.parse_args().output_dir))
