"""Pruebas confirmatorias H0-H2 del pre-registro (results/v2/fixations/PREREGISTRO.md).

Métrica primaria: FW-LPIPS. Estadístico por imagen = BD-Rate (%) por frentes de Pareto
(experiment_fixations.pair_bd); se excluye la imagen si el solape de calidad es < 50 % del rango de
la referencia.
  H0: promedio de BD(oráculo A vs. base, G_B) y BD(oráculo B vs. base, G_A); la imagen entra solo si
      las dos comparaciones son válidas (adenda del 2026-11-18).
  H1: BD(TranSalNet-Res vs. base, G).
  H2: BD(TranSalNet-Res vs. sesgo central, G), comparación directa.
Wilcoxon de rangos con signo unilateral (mediana < 0), Holm sobre H0-H2, alfa = 0,05; mediana con IC 95 %
por bootstrap (10 000, semilla 42); efecto mínimo relevante: mediana <= -2 %.
Descriptivo: H3 (costo de TranSalNet-Res: BG-LPIPS, LPIPS y PSNR globales) y exploratorio (resto).

Uso: python prereg_tests.py --split test      (validación del código: --split calib)
Salidas: results/v2/fixations/prereg_<split>.json y prereg_<split>_per_image.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiment_fixations import OUT, pair_bd

ALPHA, MIN_EFFECT, MIN_OVERLAP = 0.05, -2.0, 0.5
N_BOOT, SEED = 10_000, 42
PRIMARY_NET = "transalnet_res"


def valid(res):
    """{imagen: bd} con las comparaciones que cumplen el solape mínimo y dan un BD finito."""
    return {s: bd for s, (bd, ov) in res.items() if ov >= MIN_OVERLAP and np.isfinite(bd)}


def holm(pvals):
    """p-valores ajustados por Holm (paso descendente), en el orden de entrada."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_median_ci(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    med = np.median(x[idx], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def describe(x):
    x = np.asarray(list(x), dtype=float)
    lo, hi = bootstrap_median_ci(x)
    return {"n": int(len(x)), "median": float(np.median(x)), "ci95": [lo, hi], "improve": int((x < 0).sum())}


def confirmatory(g, eta):
    base, net, center = ("base", 0.0), (PRIMARY_NET, eta), ("center", eta)
    a = valid(pair_bd(g, base, ("oracle_A", eta), "fw_lpips_B", True))
    b = valid(pair_bd(g, base, ("oracle_B", eta), "fw_lpips_A", True))
    h0 = {s: (a[s] + b[s]) / 2 for s in sorted(set(a) & set(b))}
    h1 = valid(pair_bd(g, base, net, "fw_lpips_full", True))
    h2 = valid(pair_bd(g, center, net, "fw_lpips_full", True))
    stats = {"H0": h0, "H1": h1, "H2": h2}
    n_all = g.image.nunique()
    res = {}
    for k, d in stats.items():
        x = np.array(list(d.values()))
        r = describe(x)
        r["excluded"] = n_all - r["n"]
        r["p"] = float(wilcoxon(x, alternative="less").pvalue) if len(x) else float("nan")
        res[k] = r
    for k, p in zip(res, holm([res[k]["p"] for k in res])):
        r = res[k]
        r["p_holm"] = float(p)
        r["rejected"] = bool(p < ALPHA)
        r["verdict"] = ("efecto relevante" if r["rejected"] and r["median"] <= MIN_EFFECT else
                        "sin efecto práctico" if r["rejected"] else "no se rechaza la nula")
    if not res["H0"]["rejected"]:  # compuerta
        for k in ("H1", "H2"):
            res[k]["verdict"] += " (H0 no se rechaza: no interpretable)"
    per_image = pd.DataFrame({k: pd.Series(v) for k, v in stats.items()}).rename_axis("image")
    return res, per_image


def descriptive(g, eta):
    base, net = ("base", 0.0), (PRIMARY_NET, eta)
    h3 = {m: describe(valid(pair_bd(g, base, net, m, "lpips" in m)).values())
          for m in ["bg_lpips", "lpips", "psnr"]}
    expl = {}
    arms = sorted(set(g.arm) - {"base"})
    for arm in arms:
        gset = {"oracle_A": "B", "oracle_B": "A"}.get(arm, "full")
        for m in ["fw_lpips", "fw_psnr", "fw_ssim"]:
            v = valid(pair_bd(g, base, (arm, eta), f"{m}_{gset}", "lpips" in m)).values()
            if len(v):
                expl[f"{arm} vs base | {m}_{gset}"] = describe(v)
    v = valid(pair_bd(g, ("center", eta), ("transalnet_dense", eta), "fw_lpips_full", True)).values()
    expl["transalnet_dense vs center | fw_lpips_full"] = describe(v)
    return {"H3": h3, "exploratorio": expl}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["calib", "test"])
    ap.add_argument("--eta", type=float, default=2.0)
    args = ap.parse_args()
    g = pd.read_csv(OUT / f"grid_{args.split}.csv")
    conf, per_image = confirmatory(g, args.eta)
    out = {"split": args.split, "eta": args.eta, "n_images": int(g.image.nunique()),
           "confirmatorio": conf, **descriptive(g, args.eta)}
    (OUT / f"prereg_{args.split}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    per_image.to_csv(OUT / f"prereg_{args.split}_per_image.csv")
    print(f"split={args.split}  eta={args.eta}  imágenes={out['n_images']}")
    for k, r in conf.items():
        print(f"{k}: mediana {r['median']:+.2f} % [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
              f"mejoran {r['improve']}/{r['n']} (excl. {r['excluded']})  p={r['p']:.4f}  "
              f"p_Holm={r['p_holm']:.4f}  -> {r['verdict']}")
    for m, r in out["H3"].items():
        print(f"H3 {m}: mediana {r['median']:+.2f} % [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
