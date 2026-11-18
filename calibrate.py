"""Calibración no circular de (T_base, alpha, beta) por red de saliencia.

Objetivo: minimizar el BD-Rate en LPIPS del método guiado frente al Quadtree base, sobre el
set de calibración (CLIC 2020 validation, disjunto del benchmark; ver prepare_calibration.py).
LPIPS no conoce el mapa de saliencia (a diferencia de SW-SSIM) y el BD-Rate ya integra la
tasa, por lo que no hace falta un peso gamma*BPP.

Etapas:
  A) Baseline justo: barrido de T con alpha = beta = 0; se elige el T con mejor BD-Rate (LPIPS).
  B) Por red: Optuna/TPE (seed 42) sobre T, alpha, beta contra ese baseline.
Salidas en results/v2/calibration/.

Uso: python calibrate.py --trials 40 --nets transalnet_res u2net mobilesal transalnet_dense
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import optuna
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio

from bd_rate import bd_rate
from benchmark_datasets import SALIENCY_NETS
from src.codec import QuadtreeCodec
from src.quadtree import QuadtreeCompressor

CALIB = Path("data/calib")
OUT = Path("results/v2/calibration")
LAMBDAS = [20, 40, 150, 300, 1200]
BASE_THRESHOLDS = [0.0, 2.5, 5.0, 10.0, 20.0, 40.0]
FAIL_PENALTY = 100.0  # BD-Rate (%) asignado si las curvas no se solapan

_lpips = None


def _init():
    global _lpips
    import lpips
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    _lpips = lpips.LPIPS(net="alex", verbose=False).eval()


def _run(task):
    name, net, threshold, alpha, beta, lam, *rest = task
    normalize = bool(rest[0]) if rest else False
    scale = float(rest[1]) if len(rest) > 1 else 1.0
    img = cv2.cvtColor(cv2.imread(str(CALIB / "images" / name)), cv2.COLOR_BGR2RGB)
    if net == "uniform":  # control: peso uniforme (1 + beta) en toda la imagen, sin saliencia
        smap = np.ones(img.shape[:2], dtype=np.float32)
    else:
        smap = cv2.imread(str(CALIB / "saliency" / (net or "transalnet_res") / name),
                          cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    comp = QuadtreeCompressor(min_block_size=8, max_depth=10)
    comp.compress(img, smap, threshold=threshold, alpha=alpha, lam=lam, beta=beta,
                  normalize_weight=normalize, weight_scale=scale)
    codec = QuadtreeCodec()
    bitstream = codec.compress(comp.root, (comp.root.h, comp.root.w), comp.dynamic_quality)
    root, _, q = codec.decompress(bitstream)
    comp.orig_h, comp.orig_w = h, w
    rec = comp.reconstruct(root, override_quality=q)

    def t(x):
        return torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0)
    with torch.no_grad():
        lp = _lpips(t(img), t(rec)).item()
    return {"image": name, "lam": lam, "bpp": len(bitstream) * 8 / (h * w),
            "lpips": lp, "psnr": peak_signal_noise_ratio(img, rec, data_range=255)}


def curve(pool, images, net, threshold, alpha, beta, normalize=False):
    """Curva RD promedio sobre el set de calibración (una fila por lambda)."""
    tasks = [(n, net, threshold, alpha, beta, lam, normalize) for n in images for lam in LAMBDAS]
    df = pd.DataFrame(pool.map(_run, tasks, chunksize=1))
    return df.groupby("lam").mean(numeric_only=True).reset_index().sort_values("lam")


def bd(ref, cur, metric):
    sign = -1 if metric == "lpips" else 1
    v = bd_rate(ref.bpp.values, sign * ref[metric].values, cur.bpp.values, sign * cur[metric].values)
    return FAIL_PENALTY if np.isnan(v) else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--nets", nargs="+", default=SALIENCY_NETS)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted(p.name for p in (CALIB / "images").glob("*.png"))

    with Pool(args.workers, initializer=_init) as pool:
        # --- A) Baseline justo ---------------------------------------------------
        base_file = OUT / "baseline_sweep.csv"
        t0 = time.time()
        curves = {thr: curve(pool, images, None, thr, 0.0, 0.0) for thr in BASE_THRESHOLDS}
        ref = curves[0.0]
        rows = []
        for thr, c in curves.items():
            c.assign(threshold=thr).to_csv(OUT / f"baseline_curve_T{thr:g}.csv", index=False)
            rows.append({"threshold": thr, "bd_lpips_vs_T0": bd(ref, c, "lpips"),
                         "bd_psnr_vs_T0": bd(ref, c, "psnr"), "mean_bpp": c.bpp.mean()})
        sweep = pd.DataFrame(rows)
        sweep.to_csv(base_file, index=False)
        best_thr = float(sweep.sort_values("bd_lpips_vs_T0").threshold.iloc[0])
        base_curve = curves[best_thr]
        print(sweep.round(2).to_string(index=False))
        print(f"Baseline: T = {best_thr} ({(time.time() - t0) / 60:.1f} min)\n", flush=True)

        # --- B) Optuna por red ----------------------------------------------------
        best = {"baseline": {"threshold": best_thr, "alpha": 0.0, "beta": 0.0}}
        for net in args.nets:
            def objective(trial):
                p = {"threshold": trial.suggest_float("threshold", 0.0, 60.0),
                     "alpha": trial.suggest_float("alpha", 0.0, 20.0),
                     "beta": trial.suggest_float("beta", 0.0, 20.0)}
                c = curve(pool, images, net, p["threshold"], p["alpha"], p["beta"])
                trial.set_user_attr("bd_psnr", bd(base_curve, c, "psnr"))
                trial.set_user_attr("mean_bpp", float(c.bpp.mean()))
                return bd(base_curve, c, "lpips")

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            # Punto de partida: los valores de la memoria
            study.enqueue_trial({"threshold": 19.95, "alpha": 7.78, "beta": 4.66})
            t0 = time.time()
            study.optimize(objective, n_trials=args.trials)
            study.trials_dataframe().to_csv(OUT / f"optuna_{net}.csv", index=False)
            best[net] = study.best_params
            print(f"{net}: BD-Rate LPIPS = {study.best_value:.2f} %  params = "
                  f"{ {k: round(v, 3) for k, v in study.best_params.items()} }  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            (OUT / "best_params.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
