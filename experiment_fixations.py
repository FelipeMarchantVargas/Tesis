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

Ejecución: los procesos de CPU codifican una configuración (brazo, eta) con sus 49 puntos y calculan
PSNR/SSIM; el proceso principal calcula LPIPS (global y espacial en una sola pasada) en la GPU si hay.
Las imágenes se procesan una tras otra y cada una se escribe completa (reanudable por imagen).
"""
import os
_CUDA_ENV = os.environ.get("CUDA_VISIBLE_DEVICES")

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

# experiment_hull (vía calibrate) oculta la GPU al importarse; aquí la usa el principal para LPIPS.
if _CUDA_ENV is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_ENV

ROOT = Path("data/mit1003")
DER = ROOT / "derived"
OUT = Path("results/v2/fixations")
NETS = ["transalnet_res", "transalnet_dense", "u2net", "mobilesal"]
ARMS = NETS + ["center", "random", "oracle_A", "oracle_B"]
EVAL = {"oracle_A": "B", "oracle_B": "A"}  # el oráculo se evalúa con la otra mitad
QUALITY_COLS = ["fw_lpips", "fw_psnr", "fw_ssim"]

_ctx = {}


def _init():
    """Contexto de datos (procesos de CPU y principal). Sin LPIPS: lo calcula el principal."""
    global _ctx
    cv2.setNumThreads(1)
    cfg = json.loads((DER / "config.json").read_text())
    _ctx = {"cfg": cfg, "fix": pd.read_csv(DER / "fixations.csv"), "cb": np.load(DER / "center_bias.npy")}


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


def load_image(stem):
    return cv2.cvtColor(cv2.imread(str(ROOT / "ALLSTIMULI" / f"{stem}.jpeg")), cv2.COLOR_BGR2RGB)


def densities(stem, h, w):
    return {"full": density(stem, h, w), "A": density(stem, h, w, "A"), "B": density(stem, h, w, "B")}


def _run_config(task):
    """Proceso de CPU: las 49 codificaciones de (imagen, brazo, eta) con PSNR y SSIM ponderados.

    Devuelve las filas parciales y las reconstrucciones (uint8) para que el principal calcule LPIPS.
    """
    from skimage.metrics import peak_signal_noise_ratio
    from src.metrics import fw_psnr, fw_ssim, ssim_gray_map
    stem, arm, eta = task
    img = load_image(stem)
    h, w = img.shape[:2]
    Gs = densities(stem, h, w)
    guide = None if arm == "base" else guide_map(arm, stem, h, w)
    rows, recs = [], []
    for lam in LAMBDAS:
        for q in QUALITIES:
            rec, bpp = encode_decode(img, guide, eta, lam, q)
            sm = ssim_gray_map(img, rec)
            r = {"image": stem, "arm": arm, "eta": eta, "lam": lam, "q": q, "bpp": bpp,
                 "psnr": peak_signal_noise_ratio(img, rec, data_range=255)}
            for g, G in Gs.items():
                r[f"fw_psnr_{g}"] = fw_psnr(img, rec, G)
                r[f"fw_ssim_{g}"] = fw_ssim(img, rec, G, ssim_map=sm)
            rows.append(r)
            recs.append(rec)
    return stem, rows, recs


class LpipsScorer:
    """LPIPS en el proceso principal (GPU si hay): global, espacial ponderado por G y fondo."""

    def __init__(self, batch=7):
        import lpips
        import torch
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net="alex", verbose=False).eval().to(self.dev)
        self.batch = batch
        self._stem = None

    def _prepare(self, stem):
        from src.metrics import background_weights
        if stem != self._stem:
            img = load_image(stem)
            h, w = img.shape[:2]
            self.ref = self._t(img[None])
            dens = densities(stem, h, w)
            self.Gs = {k: self.torch.from_numpy(v).float().to(self.dev) for k, v in dens.items()}
            self.bg = self.torch.from_numpy(background_weights(dens["full"])).float().to(self.dev)
            self._stem = stem

    def _t(self, x):
        return self.torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(self.dev).float() / 127.5 - 1.0

    def score(self, stem, rows, recs):
        from src.metrics import lpips_global_and_spatial
        self._prepare(stem)
        for a in range(0, len(recs), self.batch):
            x = self._t(np.stack(recs[a:a + self.batch]))
            glob, maps = lpips_global_and_spatial(self.model, self.ref.expand(len(x), -1, -1, -1), x)
            for k, r in enumerate(rows[a:a + self.batch]):
                m = maps[k]
                r["lpips"] = float(glob[k])
                r["bg_lpips"] = float((self.bg * m).sum())
                for g, G in self.Gs.items():
                    r[f"fw_lpips_{g}"] = float((G * m).sum())
        return rows


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(DER / "split.csv")
    stems = sorted(split.loc[split.split == args.split, "image"])
    if args.limit:  # muestra al azar: los nombres de MIT1003 vienen agrupados por serie
        stems = sorted(np.random.default_rng(42).choice(stems, size=args.limit, replace=False))
    f = OUT / f"grid_{args.split}.csv"
    configs = [("base", 0.0)] + [(a, e) for a in args.arms for e in args.etas]
    expected = len(configs) * len(LAMBDAS) * len(QUALITIES)
    done = set()
    if f.exists():  # reanudación tolerante a un apagón a mitad de escritura
        g = pd.read_csv(f, on_bad_lines="skip").dropna(subset=["image", "fw_lpips_full"])
        n = g.groupby("image").size()
        complete = n[n == expected].index
        if len(complete) < len(n) or len(g) < sum(1 for _ in open(f)) - 1:
            print(f"descartando {len(n) - len(complete)} imágenes incompletas/filas dañadas", flush=True)
            g[g.image.isin(complete)].to_csv(f, index=False)
        done = set(complete)
    todo = [s for s in stems if s not in done]
    if args.max_new:  # tope de imágenes nuevas en esta ejecución (en orden)
        todo = todo[: args.max_new]
    print(f"{len(todo)} imágenes pendientes de {len(stems)}; brazos {args.arms}, eta {args.etas}", flush=True)
    tasks = [(s, a, e) for s in todo for a, e in configs]  # imagen por imagen
    pending = {s: [] for s in todo}
    cols = None
    if f.exists():
        cols = list(pd.read_csv(f, nrows=0).columns)
    t0, n_done = time.time(), 0
    # El Pool se crea antes de tocar CUDA en el principal (los procesos hijos nacen por fork).
    with Pool(args.workers or os.cpu_count(), initializer=_init) as pool:
        _init()
        scorer = LpipsScorer()
        print(f"LPIPS en {scorer.dev}", flush=True)
        for stem, rows, recs in pool.imap_unordered(_run_config, tasks):
            pending[stem] += scorer.score(stem, rows, recs)
            if len(pending[stem]) == expected:
                df = pd.DataFrame(pending.pop(stem))
                df = df[cols] if cols else df
                header = not f.exists()
                with open(f, "a") as fh:  # una escritura por imagen, forzada a disco
                    df.to_csv(fh, header=header, index=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                cols = list(df.columns)
                n_done += 1
                el = time.time() - t0
                print(f"  {n_done}/{len(todo)}  {stem}  {el / n_done:.0f} s/imagen  "
                      f"(restan ~{el / n_done * (len(todo) - n_done) / 60:.0f} min)", flush=True)


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
    r.add_argument("--max-new", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("--split", default="calib", choices=["calib", "test"])
    a.add_argument("--min-overlap", type=float, default=0.5)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else analyze(args)
