"""Registro de datasets del benchmark (Memoria §4.1.1; CLIC = test sets 2020 Professional + Mobile)."""
from pathlib import Path

DATASETS = {
    "kodak": ("data/kodak", "*.png"),
    "div2k": ("data/div2k_val/DIV2K_valid_HR", "*.png"),
    "urban100": ("data/urban100/Urban100_HR", "*.png"),
    "clic_professional": ("data/clic/CLIC2020Professional_test", "**/*.png"),
    "clic_mobile": ("data/clic/CLIC2020Mobile_test", "**/*.png"),
}
# Datasets con fijaciones humanas: solo para el protocolo de evaluación, fuera del benchmark.
FIXATION_DATASETS = {
    "mit1003": ("data/mit1003/ALLSTIMULI", "*.jpeg"),
}
SALIENCY_NETS = ["u2net", "mobilesal", "transalnet_dense", "transalnet_res"]
SALIENCY_DIR = Path("data/saliency")


def list_images(dataset):
    root, pattern = {**DATASETS, **FIXATION_DATASETS}[dataset]
    return sorted(Path(root).glob(pattern))


def saliency_path(net, dataset, image_path):
    return SALIENCY_DIR / net / dataset / (Path(image_path).stem + ".png")
