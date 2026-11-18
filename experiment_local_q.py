"""Experimento B: cuantización local guiada por saliencia (qmap), evaluada sobre envolventes de Pareto.

Mismo protocolo que experiment_hull.py (grilla lambda x Q, frente de Pareto por imagen, BD-Rate por
interpolación, mediana por imagen) contra el baseline sin qmap. RDO con MSE puro (beta = 0): la
saliencia actúa SOLO a través de la cuantización por celda (32x32, 4 niveles, en el bitstream).

Controles sin saliencia (misma intensidad eta):
  - random: mapa de ruido uniforme suavizado (variar la cuantización sin información);
  - center: sesgo central gaussiano (predictor trivial de fijaciones).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import calibrate as cal
from experiment_hull import LAMBDAS, QUALITIES, bd_rate_interp, pareto

OUT = Path("results/v2/local_q")
CONFIGS = {  # nombre: (mapa, eta)
    "tsnres_eta1": ("transalnet_res", 1.0),
    "tsnres_eta2": ("transalnet_res", 2.0),
    "tsnres_eta3": ("transalnet_res", 3.0),
    "u2net_eta2": ("u2net", 2.0),
    "center_eta2": ("center", 2.0),
    "random_eta2": ("random", 2.0),
}


def load_map(kind, name, shape):
    h, w = shape
    if kind == "center":
        yy, xx = np.mgrid[-1:1:h * 1j, -1:1:w * 1j]
        return np.exp(-(xx ** 2 + yy ** 2) / (2 * 0.4 ** 2)).astype(np.float32)
    if kind == "random":
        rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
        m = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 24)
        return (m - m.min()) / (m.max() - m.min())
    return cv2.imread(str(cal.CALIB / "saliency" / kind / name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def _run(task):
    name, kind, eta, lam, q = task
    import torch
    from skimage.metrics import peak_signal_noise_ratio
    from src.codec import QuadtreeCodec
    from src.quadtree import QuadtreeCompressor
    img = cv2.cvtColor(cv2.imread(str(cal.CALIB / "images" / name)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    smap = load_map(kind, name, (h, w))
    comp = QuadtreeCompressor(8, 10)
    comp.compress(img, smap, 0.0, 0.0, lam=lam, beta=0.0, quality=q, qmap_strength=eta)
    codec = QuadtreeCodec()
    bs = codec.compress(comp.root, (comp.root.h, comp.root.w), comp.dynamic_quality,
                        qmap=comp.qmap, qmap_strength=comp.qmap_strength)
    root, _, qq = codec.decompress(bs)
    dec = QuadtreeCompressor(8, 10)
    dec.orig_h, dec.orig_w = h, w
    dec.qmap, dec.qmap_strength = codec.qmap, codec.qmap_strength
    dec.qmap_cell, dec.qmap_levels = codec.qmap_cell, codec.qmap_levels
    rec = dec.reconstruct(root, override_quality=qq)

    def t(x):
        return torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0)
    with torch.no_grad():
        lp = cal._lpips(t(img), t(rec)).item()
    return {"image": name, "lam": lam, "q": q, "bpp": len(bs) * 8 / (h * w), "lpips": lp,
            "psnr": peak_signal_noise_ratio(img, rec, data_range=255)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted(p.name for p in (cal.CALIB / "images").glob("*.png"))
    base = pd.read_csv("results/v2/hull/grid_base.csv")
    rows = []
    with Pool(os.cpu_count(), initializer=cal._init) as pool:
        for cfg, (kind, eta) in CONFIGS.items():
            f = OUT / f"grid_{cfg}.csv"
            if not f.exists():
                tasks = [(n, kind, eta, lam, q) for n in images for lam in LAMBDAS for q in QUALITIES]
                pd.DataFrame(pool.map(_run, tasks, chunksize=2)).assign(config=cfg).to_csv(f, index=False)
            g = pd.read_csv(f)
            for metric in ["lpips", "psnr"]:
                v = np.array([bd_rate_interp(*pareto(base[base.image == n], metric),
                                             *pareto(g[g.image == n], metric)) for n in images])
                rows.append({"config": cfg, "metric": metric, "bd_median": np.nanmedian(v),
                             "bd_mean": np.nanmean(v), "improve": int(np.sum(v < 0)), "n": int(np.sum(~np.isnan(v)))})
                print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT / "local_q_bdrate.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
