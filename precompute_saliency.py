"""Precalcula los mapas de saliencia de las 4 redes para todos los datasets (GPU).

Los mapas se guardan como PNG uint8 normalizados [0, 255] en data/saliency/<red>/<dataset>/.
Es reanudable: omite los mapas ya existentes.
"""
import argparse
import time

import cv2
import numpy as np
from tqdm import tqdm

from benchmark_datasets import DATASETS, FIXATION_DATASETS, SALIENCY_NETS, list_images, saliency_path
from src.saliency import get_saliency_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS) + list(FIXATION_DATASETS))
    args = ap.parse_args()
    for net in SALIENCY_NETS:
        jobs = [(d, p) for d in args.datasets for p in list_images(d)
                if not saliency_path(net, d, p).exists()]
        if not jobs:
            continue
        model = get_saliency_model(net)
        t0 = time.time()
        for dataset, path in tqdm(jobs, desc=net):
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            smap = model.get_saliency_map(img)
            out = saliency_path(net, dataset, path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.round(smap * 255).astype(np.uint8))
        print(f"{net}: {len(jobs)} mapas en {time.time() - t0:.0f} s")
        del model
