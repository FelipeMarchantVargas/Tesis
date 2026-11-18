"""Diagnóstico de la calibración (results/v2/calibration/).

Genera:
  - baseline_sweep.png: BD-Rate (LPIPS y PSNR) del baseline vs T (referencia T = 0).
  - sensitivity_<red>.png: objetivo de cada trial vs T, alpha y beta (bordes = líneas rojas).
  - rd_calibration.png: curvas RD en el set de calibración: baseline calibrado, parámetros de
    la memoria y parámetros calibrados de cada red (recalcula las curvas, ~1 min por curva).
"""
import json
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import calibrate as cal

OUT = cal.OUT
MEMORIA = {"threshold": 19.95, "alpha": 7.78, "beta": 4.66}
RANGES = {"threshold": 60.0, "alpha": 20.0, "beta": 20.0}


def plot_sweep():
    s = pd.read_csv(OUT / "baseline_sweep.csv")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(s.threshold, s.bd_lpips_vs_T0, "o-", label="BD-Rate LPIPS")
    ax.plot(s.threshold, s.bd_psnr_vs_T0, "s--", label="BD-Rate PSNR")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("T (baseline, α = β = 0)")
    ax.set_ylabel("BD-Rate vs T = 0 (%)")
    ax.set_title("Baseline: efecto del umbral top-down")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "baseline_sweep.png", dpi=120)


def plot_sensitivity(net):
    df = pd.read_csv(OUT / f"optuna_{net}.csv")
    df = df[df.state == "COMPLETE"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, p in zip(axes, ["threshold", "alpha", "beta"]):
        sc = ax.scatter(df[f"params_{p}"], df.value, c=df.number, cmap="viridis", s=25)
        best = df.loc[df.value.idxmin()]
        ax.scatter(best[f"params_{p}"], best.value, marker="*", s=250, c="red", zorder=3)
        ax.axvline(RANGES[p], color="red", ls=":", lw=1)
        ax.axvline(MEMORIA[p], color="gray", ls="--", lw=1)
        ax.set_xlabel(p)
    axes[0].set_ylabel("BD-Rate LPIPS vs baseline (%)")
    fig.colorbar(sc, ax=axes, label="trial")
    fig.suptitle(f"{net}: sensibilidad (★ óptimo, -- memoria, ··· borde del rango)")
    fig.savefig(OUT / f"sensitivity_{net}.png", dpi=120, bbox_inches="tight")


def plot_rd():
    best = json.loads((OUT / "best_params.json").read_text())
    images = sorted(p.name for p in (cal.CALIB / "images").glob("*.png"))
    base = best["baseline"]
    nets = [n for n in best if n != "baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    with Pool(os.cpu_count(), initializer=cal._init) as pool:
        curves = {"baseline (T={:g})".format(base["threshold"]):
                  cal.curve(pool, images, None, base["threshold"], 0, 0)}
        for net in nets:
            curves[f"{net} memoria"] = cal.curve(pool, images, net, **MEMORIA)
            curves[f"{net} calibrado"] = cal.curve(pool, images, net, **best[net])
    rows = []
    ref = next(iter(curves.values()))
    for label, c in curves.items():
        style = "k-" if label.startswith("baseline") else ("--" if "memoria" in label else "-")
        for ax, m in zip(axes, ["lpips", "psnr"]):
            ax.plot(c.bpp, c[m], style, marker="o", ms=3, label=label)
        rows.append({"curva": label, "bd_lpips": cal.bd(ref, c, "lpips"), "bd_psnr": cal.bd(ref, c, "psnr")})
    axes[0].invert_yaxis()
    axes[0].set_ylabel("LPIPS (invertido)")
    axes[1].set_ylabel("PSNR (dB)")
    for ax in axes:
        ax.set_xlabel("BPP")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle("Set de calibración (CLIC 2020 validation, 20 recortes 768²)")
    fig.tight_layout()
    fig.savefig(OUT / "rd_calibration.png", dpi=120)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "rd_calibration_bdrate.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    plot_sweep()
    best = json.loads((OUT / "best_params.json").read_text())
    for net in best:
        if net != "baseline":
            plot_sensitivity(net)
    plot_rd()
