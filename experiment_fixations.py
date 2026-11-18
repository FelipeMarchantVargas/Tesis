"""Protocolo con fijaciones reales (MIT1003): cuantización local guiada, evaluada donde miran personas.

Ver la nota «Protocolo de evaluación con fijaciones reales» (bóveda de la memoria). Mecanismo: qmap del
Exp. B (RDO con MSE, beta = 0). Cada mapa guía se normaliza por rangos a nivel de celda
(src/fixations.py: rank_normalize_cells), así que los brazos solo difieren en *dónde* ponen los bits.

Brazos: base (eta = 0), 4 redes, sesgo central empírico, aleatorio, y oráculo cruzado (guía = fijaciones
de la mitad A de los observadores; se evalúa con la mitad B, y viceversa).
Cada codificación se mide con G de todos los observadores (full), de la mitad A y de la mitad B:
FW-LPIPS, FW-PSNR, FW-SSIM, BG-LPIPS (fondo, G = todos), más LPIPS y PSNR globales.

Uso:
  python experiment_fixations.py run --split calib --limit 30 --etas 1 2     # piloto
  python experiment_fixations.py analyze --split calib
Salidas en results/v2/fixations/.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from benchmark_datasets import saliency_path
from experiment_hull import LAMBDAS, QUALITIES, bd_rate_interp, pareto
from prepare_fixations import points
from src.fixations import fixation_density, rank_normalize_cells

ROOT = Path("data/mit1003")
DER = ROOT / "derived"
OUT = Path("results/v2/fixations")
NETS = ["transalnet_res", "transalnet_dense", "u2net", "mobilesal"]
ARMS = NETS + ["center", "random", "oracle_A", "oracle_B"]
EVAL = {"oracle_A": "B", "oracle_B": "A"}  # el oráculo se evalúa con la otra mitad
QUALITY_COLS = ["fw_lpips", "fw_psnr", "fw_ssim"]

_ctx = {}


def _init():
    global _ctx
    import lpips
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    cfg = json.loads((DER / "config.json").read_text())
    _ctx = {"cfg": cfg, "fix": pd.read_csv(DER / "fixations.csv"), "cb": np.load(DER / "center_bias.npy"),
            "lp_spatial": lpips.LPIPS(net="alex", spatial=True, verbose=False).eval(),
            "lp": lpips.LPIPS(net="alex", verbose=False).eval()}


def density(stem, h, w, half=None):
    fx = _ctx["fix"][_ctx["fix"].image == stem]
    if half is not None:
        fx = fx[fx.half == half]
    cfg = _ctx["cfg"]
    return fixation_density(points(fx, w, h, cfg["offset"], cfg["skip_first"]), (h, w), sigma=cfg["sigma_px"])


def guide_map(arm, stem, h, w):
    if arm in NETS:
        m = cv2.imread(str(saliency_path(arm, "mit1003", ROOT / "ALLSTIMULI" / f"{stem}.jpeg")), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"falta saliencia {arm}/{stem}: correr precompute_saliency.py --datasets mit1003")
        m = cv2.resize(m.astype(np.float32) / 255.0, (w, h))
    elif arm == "center":
        m = cv2.resize(_ctx["cb"].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    elif arm == "random":  # como en el Exp. B, pero con semilla estable entre corridas
        rng = np.random.default_rng(zlib.crc32(stem.encode()))
        m = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 24)
    elif arm.startswith("oracle_"):
        m = density(stem, h, w, half=arm[-1]).astype(np.float32)
    else:
        raise ValueError(arm)
    return rank_normalize_cells(m, cell=32)


def encode_decode(img, guide, eta, lam, q):
    from src.codec import QuadtreeCodec
    from src.quadtree import QuadtreeCompressor
    h, w = img.shape[:2]
    smap = guide if guide is not None else np.ones((h, w), np.float32)
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
    return dec.reconstruct(root, override_quality=qq), len(bs) * 8 / (h * w)


def _run_image(task):
    """Todas las codificaciones (brazos x eta x lambda x Q) de una imagen."""
    import torch
    from skimage.metrics import peak_signal_noise_ratio
    from src.metrics import background_weights, fw_lpips, fw_psnr, fw_ssim, lpips_spatial_map
    stem, arms, etas = task
    img = cv2.cvtColor(cv2.imread(str(ROOT / "ALLSTIMULI" / f"{stem}.jpeg")), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    Gs = {"full": density(stem, h, w), "A": density(stem, h, w, "A"), "B": density(stem, h, w, "B")}
    bg = background_weights(Gs["full"])

    def t(x):
        return torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0)

    rows = []
    for arm, eta in [("base", 0.0)] + [(a, e) for a in arms for e in etas]:
        guide = None if arm == "base" else guide_map(arm, stem, h, w)
        for lam in LAMBDAS:
            for q in QUALITIES:
                rec, bpp = encode_decode(img, guide, eta, lam, q)
                lm = lpips_spatial_map(_ctx["lp_spatial"], img, rec)
                with torch.no_grad():
                    lp = _ctx["lp"](t(img), t(rec)).item()
                r = {"image": stem, "arm": arm, "eta": eta, "lam": lam, "q": q, "bpp": bpp,
                     "lpips": lp, "psnr": peak_signal_noise_ratio(img, rec, data_range=255),
                     "bg_lpips": fw_lpips(lm, bg)}
                for g, G in Gs.items():
                    r[f"fw_lpips_{g}"] = fw_lpips(lm, G)
                    r[f"fw_psnr_{g}"] = fw_psnr(img, rec, G)
                    r[f"fw_ssim_{g}"] = fw_ssim(img, rec, G)
                rows.append(r)
    return rows


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(DER / "split.csv")
    stems = sorted(split.loc[split.split == args.split, "image"])
    if args.limit:  # muestra al azar: los nombres de MIT1003 vienen agrupados por serie
        stems = sorted(np.random.default_rng(42).choice(stems, size=args.limit, replace=False))
    f = OUT / f"grid_{args.split}.csv"
    done = set(pd.read_csv(f).image) if f.exists() else set()
    todo = [s for s in stems if s not in done]
    print(f"{len(todo)} imágenes pendientes de {len(stems)}; brazos {args.arms}, eta {args.etas}", flush=True)
    t0 = time.time()
    with Pool(args.workers or os.cpu_count(), initializer=_init) as pool:
        for i, rows in enumerate(pool.imap_unordered(_run_image, [(s, args.arms, args.etas) for s in todo]), 1):
            pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el / i:.0f} s/imagen  (restan ~{el / i * (len(todo) - i) / 60:.0f} min)", flush=True)


def per_image_bd(g, arm, eta, metric):
    """BD-Rate por imagen de (arm, eta) frente a base sobre la métrica `metric` (columna base, sin sufijo)."""
    gset = EVAL.get(arm, "full")
    col = f"{metric}_{gset}" if metric in QUALITY_COLS else metric
    lower_better = "lpips" in metric
    out = []
    for stem, d in g.groupby("image"):
        a = d[d.arm == "base"].assign(qual=lambda x: -x[col] if lower_better else x[col])
        b = d[(d.arm == arm) & (d.eta == eta)].assign(qual=lambda x: -x[col] if lower_better else x[col])
        if a.empty or b.empty:
            continue
        ra, qa = pareto(a, "qual")
        rb, qb = pareto(b, "qual")
        lo, hi = max(qa.min(), qb.min()), min(qa.max(), qb.max())
        overlap = max(0.0, hi - lo) / (qa.max() - qa.min()) if qa.max() > qa.min() else 0.0
        out.append({"image": stem, "arm": arm, "eta": eta, "metric": metric, "eval_g": gset,
                    "bd": bd_rate_interp(ra, qa, rb, qb), "overlap": overlap})
    return out


def analyze(args):
    g = pd.read_csv(OUT / f"grid_{args.split}.csv")
    rows = []
    for (arm, eta), _ in g[g.arm != "base"].groupby(["arm", "eta"]):
        for metric in QUALITY_COLS + ["bg_lpips", "lpips", "psnr"]:
            rows += per_image_bd(g, arm, eta, metric)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / f"bd_per_image_{args.split}.csv", index=False)
    ok = per[per.overlap >= args.min_overlap]
    summ = ok.groupby(["metric", "arm", "eta"]).bd.agg(
        median="median", improve=lambda v: int((v < 0).sum()), n="count").reset_index()
    summ["excluded"] = summ.apply(lambda r: int(((per.metric == r.metric) & (per.arm == r.arm) & (per.eta == r.eta)).sum()) - r.n, axis=1)
    summ.to_csv(OUT / f"bd_summary_{args.split}.csv", index=False)
    print(summ[summ.metric == "fw_lpips"].round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", default="calib", choices=["calib", "test"])
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    r.add_argument("--etas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    r.add_argument("--workers", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("--split", default="calib", choices=["calib", "test"])
    a.add_argument("--min-overlap", type=float, default=0.5)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else analyze(args)
