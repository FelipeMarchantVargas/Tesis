"""Barrido explícito de beta con T = 0 (árbol completo; alpha no influye) para cada red.

Motivo: la calibración con Optuna (calibrate.py, objetivo BD-Rate LPIPS no circular) llevó
T -> 0, y con T = 0 el split top-down divide todo hasta 8x8, por lo que alpha deja de tener
efecto. Queda un único parámetro, beta: un barrido 1D muestra la curva completa (incluida la
saturación) en vez de solo un óptimo en el borde del rango.

Salidas: results/v2/calibration/beta_sweep.csv y beta_sweep.png
"""
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import calibrate as cal
from benchmark_datasets import SALIENCY_NETS

BETAS = [0, 1, 2, 5, 10, 20, 40, 80, 160]

if __name__ == "__main__":
    images = sorted(p.name for p in (cal.CALIB / "images").glob("*.png"))
    rows = []
    with Pool(os.cpu_count(), initializer=cal._init) as pool:
        base = cal.curve(pool, images, None, 0.0, 0.0, 0.0)
        for net in SALIENCY_NETS:
            for beta in BETAS:
                c = cal.curve(pool, images, net, 0.0, 0.0, float(beta))
                c.assign(net=net, beta=beta).to_csv(cal.OUT / f"beta_curve_{net}_b{beta}.csv", index=False)
                rows.append({"net": net, "beta": beta, "bd_lpips": cal.bd(base, c, "lpips"),
                             "bd_psnr": cal.bd(base, c, "psnr"), "mean_bpp": c.bpp.mean()})
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(cal.OUT / "beta_sweep.csv", index=False)
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for net, d in df.groupby("net"):
        axes[0].plot(d.beta + 0.5, d.bd_lpips, "o-", label=net)
        axes[1].plot(d.beta + 0.5, d.bd_psnr, "o-", label=net)
    for ax, t in zip(axes, ["BD-Rate LPIPS (%)", "BD-Rate PSNR (%)"]):
        ax.set_xscale("log")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("β + 0,5 (log)")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Calibración: β con T = 0, vs baseline T = 0 (CLIC 2020 validation)")
    fig.tight_layout()
    fig.savefig(cal.OUT / "beta_sweep.png", dpi=120)
    print(df.pivot(index="beta", columns="net", values="bd_lpips").round(2))
