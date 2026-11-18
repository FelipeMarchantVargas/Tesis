"""Experimento A: comparación sobre la envolvente de Pareto en una grilla (lambda, Q).

El códec original ata la calidad DCT a lambda (Q = 100 - 12 ln(lambda + 1)); ese acoplamiento
contamina la comparación con saliencia (ver results/v2/calibration/INFORME.md). Aquí lambda y Q
se recorren por separado y cada método se representa por su frente de Pareto por imagen
(BPP vs calidad), de modo que ningún mapeo lambda -> Q favorece a un método.

BD-Rate entre frentes: interpolación lineal de log(BPP) en función de la calidad sobre el
intervalo común (robusto con muchos puntos). Se reporta la mediana por imagen.

Uso: python experiment_hull.py [--configs base tsnres_b4 ...]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate as cal

OUT = Path("results/v2/hull")
LAMBDAS = [0.1, 0.5, 2, 10, 50, 250, 1250]
QUALITIES = [10, 20, 30, 45, 60, 75, 90]
CONFIGS = {  # nombre: (red, beta) ; peso normalizado (media 1) -> lambda conserva su escala
    "base": (None, 0.0),
    "tsnres_b4": ("transalnet_res", 4.0),
    "tsnres_b16": ("transalnet_res", 16.0),
    "u2net_b16": ("u2net", 16.0),
}


def _run(task):
    name, net, beta, lam, q = task
    import cv2
    import torch
    from skimage.metrics import peak_signal_noise_ratio
    from src.codec import QuadtreeCodec
    from src.quadtree import QuadtreeCompressor
    img = cv2.cvtColor(cv2.imread(str(cal.CALIB / "images" / name)), cv2.COLOR_BGR2RGB)
    smap = cv2.imread(str(cal.CALIB / "saliency" / (net or "transalnet_res") / name),
                      cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    comp = QuadtreeCompressor(min_block_size=8, max_depth=10)
    comp.compress(img, smap, threshold=0.0, alpha=0.0, lam=lam, beta=beta,
                  normalize_weight=True, quality=q)
    codec = QuadtreeCodec()
    bs = codec.compress(comp.root, (comp.root.h, comp.root.w), comp.dynamic_quality)
    root, _, qq = codec.decompress(bs)
    comp.orig_h, comp.orig_w = h, w
    rec = comp.reconstruct(root, override_quality=qq)

    def t(x):
        return torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0)
    with torch.no_grad():
        lp = cal._lpips(t(img), t(rec)).item()
    return {"image": name, "lam": lam, "q": q, "bpp": len(bs) * 8 / (h * w), "lpips": lp,
            "psnr": peak_signal_noise_ratio(img, rec, data_range=255)}


def pareto(df, metric):
    """Frente de Pareto (BPP mínima para cada nivel de calidad). Devuelve (bpp, calidad) ordenados."""
    qual = -df[metric].values if metric == "lpips" else df[metric].values
    order = np.argsort(df.bpp.values)
    r, qv, best = [], [], -np.inf
    for i in order:
        if qual[i] > best:
            best = qual[i]
            r.append(df.bpp.values[i])
            qv.append(qual[i])
    return np.array(r), np.array(qv)


def bd_rate_interp(r1, q1, r2, q2, n=200):
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo:
        return np.nan
    grid = np.linspace(lo, hi, n)
    d = np.interp(grid, q2, np.log(r2)) - np.interp(grid, q1, np.log(r1))
    return (np.exp(d.mean()) - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted(p.name for p in (cal.CALIB / "images").glob("*.png"))
    with Pool(os.cpu_count(), initializer=cal._init) as pool:
        for cfg in args.configs:
            f = OUT / f"grid_{cfg}.csv"
            if f.exists():
                continue
            net, beta = CONFIGS[cfg]
            tasks = [(n, net, beta, lam, q) for n in images for lam in LAMBDAS for q in QUALITIES]
            pd.DataFrame(pool.map(_run, tasks, chunksize=2)).assign(config=cfg).to_csv(f, index=False)
            print(f"{cfg}: listo", flush=True)
    grids = {c: pd.read_csv(OUT / f"grid_{c}.csv") for c in CONFIGS if (OUT / f"grid_{c}.csv").exists()}
    rows = []
    for cfg, g in grids.items():
        if cfg == "base":
            continue
        for metric in ["lpips", "psnr"]:
            v = []
            for n in images:
                a, b = grids["base"][grids["base"].image == n], g[g.image == n]
                v.append(bd_rate_interp(*pareto(a, metric), *pareto(b, metric)))
            v = np.array(v)
            rows.append({"config": cfg, "metric": metric, "bd_median": np.nanmedian(v),
                         "bd_mean": np.nanmean(v), "improve": int(np.sum(v < 0)), "n": int(np.sum(~np.isnan(v)))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hull_bdrate.csv", index=False)
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
