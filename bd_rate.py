"""BD-Rate (Bjøntegaard, VCEG-M33) del benchmark v2 respecto al Quadtree base.

Se calcula sobre la curva promedio de cada dataset (BPP y métrica promediados por lambda),
con ajuste polinomial cúbico de log(R) en función de la calidad, integrado en el intervalo
de calidad común. Para LPIPS se usa -LPIPS como calidad (mayor es mejor).
También reporta la mediana del BD-Rate por imagen y si las curvas son monótonas.

Uso: python bd_rate.py results/v2/benchmark.csv
"""
import sys

import numpy as np
import pandas as pd

METRICS = {"psnr": 1, "ssim": 1, "sw_ssim": 1, "lpips": -1}


def bd_rate(r1, q1, r2, q2):
    """Diferencia media de tasa (%) de la curva 2 respecto de la 1. Negativo = ahorro."""
    lr1, lr2 = np.log(r1), np.log(r2)
    lo, hi = max(min(q1), min(q2)), min(max(q1), max(q2))
    if hi <= lo:
        return np.nan
    p1, p2 = np.polyfit(q1, lr1, 3), np.polyfit(q2, lr2, 3)
    i1 = np.polyval(np.polyint(p1), hi) - np.polyval(np.polyint(p1), lo)
    i2 = np.polyval(np.polyint(p2), hi) - np.polyval(np.polyint(p2), lo)
    return (np.exp((i2 - i1) / (hi - lo)) - 1) * 100


def monotone(r, q):
    order = np.argsort(r)
    return bool(np.all(np.diff(np.asarray(q)[order]) > 0))


def main(path):
    df = pd.read_csv(path)
    qt = df[~df.method.isin(["jpeg", "webp"])]
    rows = []
    for dataset, d in qt.groupby("dataset"):
        mean = d.groupby(["method", "param"]).mean(numeric_only=True).reset_index()
        ref = mean[mean.method == "base"].sort_values("param")
        for method in [m for m in mean.method.unique() if m != "base"]:
            cur = mean[mean.method == method].sort_values("param")
            row = {"dataset": dataset, "method": method}
            for metric, sign in METRICS.items():
                row[f"bd_{metric}"] = bd_rate(ref.bpp.values, sign * ref[metric].values,
                                              cur.bpp.values, sign * cur[metric].values)
                per_image = []
                for img, di in d.groupby("image"):
                    a = di[di.method == "base"].sort_values("param")
                    b = di[di.method == method].sort_values("param")
                    if len(a) > 3 and len(b) > 3:
                        per_image.append(bd_rate(a.bpp.values, sign * a[metric].values,
                                                 b.bpp.values, sign * b[metric].values))
                row[f"bd_{metric}_median_img"] = np.nanmedian(per_image) if per_image else np.nan
                row[f"monotone_{metric}"] = (monotone(ref.bpp, sign * ref[metric])
                                             and monotone(cur.bpp, sign * cur[metric]))
            rows.append(row)
    res = pd.DataFrame(rows)
    out = path.replace(".csv", "_bdrate.csv")
    res.to_csv(out, index=False)
    cols = ["dataset", "method"] + [f"bd_{m}" for m in METRICS] + ["bd_lpips_median_img", "monotone_lpips"]
    print(res[cols].round(2).to_string(index=False))
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/v2/benchmark.csv")
