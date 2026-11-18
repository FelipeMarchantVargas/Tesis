"""Set de calibración disjunto del benchmark: CLIC 2020 *validation* (Professional + Mobile).

Toma N imágenes por subconjunto (semilla fija) con lado menor >= CROP y guarda un recorte
central CROP x CROP. La saliencia se calcula sobre la imagen completa y luego se recorta.
Salida: data/calib/images/*.png y data/calib/saliency/<red>/*.png
"""
import random
from pathlib import Path

import cv2
import numpy as np

from benchmark_datasets import SALIENCY_NETS
from src.saliency import get_saliency_model

N_PER_SET, CROP, SEED = 10, 768, 42
OUT = Path("data/calib")


def center_crop(a, size):
    h, w = a.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return a[y:y + size, x:x + size]


if __name__ == "__main__":
    rng = random.Random(SEED)
    chosen = []
    for subset in ["professional_valid_2020", "mobile_valid_2020"]:
        files = []
        for p in sorted(Path(f"data/clic/{subset}").glob("**/*.png")):
            img = cv2.imread(str(p))  # omite archivos ilegibles (p. ej. metadatos ._ de macOS)
            if img is not None and min(img.shape[:2]) >= CROP:
                files.append(p)
        chosen += [(subset.split("_")[0], p) for p in rng.sample(files, N_PER_SET)]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    full = {}
    for tag, p in chosen:
        img = cv2.imread(str(p))
        name = f"{tag}_{p.stem}.png"
        cv2.imwrite(str(OUT / "images" / name), center_crop(img, CROP))
        full[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for net in SALIENCY_NETS:
        model = get_saliency_model(net)
        (OUT / "saliency" / net).mkdir(parents=True, exist_ok=True)
        for name, img in full.items():
            smap = center_crop(model.get_saliency_map(img), CROP)
            cv2.imwrite(str(OUT / "saliency" / net / name), np.round(smap * 255).astype(np.uint8))
        del model
    print(f"{len(full)} imágenes de calibración en {OUT}")
