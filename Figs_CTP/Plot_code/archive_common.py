"""Common paths and publication exports."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"Plot_data"

def export(fig,number,output_dir=None,main=False):
    folder=Path(output_dir) if output_dir else ROOT/"Reproduced"/("Main_figs" if main else "Supp_figs")
    folder.mkdir(parents=True,exist_ok=True)
    stem=folder/f"{'main' if main else 'supp'}_figure_{number}"
    fig.savefig(stem.with_suffix(".png"),dpi=600,facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"),dpi=600,facecolor="white")
    fig.savefig(stem.with_suffix(".svg"),dpi=600,facecolor="white")
    with Image.open(stem.with_suffix(".png")) as image:
        image.convert("RGB").save(stem.with_suffix(".tiff"),dpi=(600,600),compression="tiff_lzw")
    plt.close(fig)
    return stem
