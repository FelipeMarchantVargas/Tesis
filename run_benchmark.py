"""Benchmark v2 (post-fix a860e86): Quadtree base vs SG-RDO-QT con 4 redes de saliencia, JPEG y WebP.

- Corre en CPU con N procesos; los mapas de saliencia vienen de precompute_saliency.py.
- Reanudable: omite las combinaciones (dataset, imagen, método) ya presentes en el CSV.
- Métricas: BPP, PSNR, SSIM, SW-SSIM (mapa de referencia común: TranSalNet-Res), LPIPS (AlexNet).
- Los tiempos de codificación NO incluyen la inferencia de saliencia (precalculada).

Uso: python run_benchmark.py --params results/v2/params.json --workers 8
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # todo en CPU; la GPU solo se usa para precalcular saliencia

import argparse
import csv
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from benchmark_datasets import DATASETS, SALIENCY_NETS, list_images, saliency_path
from src.codec import QuadtreeCodec
from src.quadtree import QuadtreeCompressor

LAMBDAS = [10, 20, 40, 80, 150, 300, 600, 1200]
QUALITIES = [5, 10, 20, 30, 50, 70, 90]
REFERENCE_MAP = "transalnet_res"
FIELDS = ["dataset", "image", "method", "param", "width", "height", "bpp", "psnr", "ssim",
          "sw_ssim", "lpips", "enc_time", "dec_time", "leaves_flat", "leaves_interp", "leaves_dct"]

_lpips = None
_params = None


def _init(params):
    global _lpips, _params
    import lpips
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    _lpips = lpips.LPIPS(net="alex", verbose=False).eval()
    _params = params


def _load_map(net, dataset, path):
    return cv2.imread(str(saliency_path(net, dataset, path)), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def _metrics(orig, rec, ref_map):
    ssim_val, ssim_map = structural_similarity(orig, rec, data_range=255, channel_axis=2, full=True)
    ssim_gray = ssim_map.mean(axis=2)
    sw = float((ssim_gray * ref_map).sum() / ref_map.sum()) if ref_map.sum() > 0 else ssim_val

    def t(x):
        return torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0)
    with torch.no_grad():
        lp = _lpips(t(orig), t(rec)).item()
    return {"psnr": peak_signal_noise_ratio(orig, rec, data_range=255), "ssim": ssim_val,
            "sw_ssim": sw, "lpips": lp}


def _leaf_modes(node, acc):
    if node.children:
        for child in node.children:
            _leaf_modes(child, acc)
    else:
        acc[node.mode] += 1
    return acc


def run_task(task):
    dataset, path, method = task
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    ref_map = _load_map(REFERENCE_MAP, dataset, path)
    rows = []
    base = {"dataset": dataset, "image": Path(path).name, "method": method, "width": w, "height": h}

    if method in ("jpeg", "webp"):
        flag = cv2.IMWRITE_JPEG_QUALITY if method == "jpeg" else cv2.IMWRITE_WEBP_QUALITY
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for q in QUALITIES:
            t0 = time.perf_counter()
            ok, enc = cv2.imencode("." + ("jpg" if method == "jpeg" else "webp"), bgr, [flag, q])
            t1 = time.perf_counter()
            rec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            t2 = time.perf_counter()
            rows.append({**base, "param": q, "bpp": len(enc) * 8 / (h * w), "enc_time": t1 - t0,
                         "dec_time": t2 - t1, **_metrics(img, rec, ref_map)})
        return rows

    if method == "base":
        smap, alpha, beta = ref_map, 0.0, 0.0  # con alpha = beta = 0 el mapa no influye
    else:
        smap, alpha, beta = _load_map(method, dataset, path), _params["alpha"], _params["beta"]
    codec = QuadtreeCodec()
    for lam in LAMBDAS:
        comp = QuadtreeCompressor(min_block_size=8, max_depth=10)
        t0 = time.perf_counter()
        comp.compress(img, smap, threshold=_params["threshold"], alpha=alpha, lam=lam, beta=beta)
        bitstream = codec.compress(comp.root, (comp.root.h, comp.root.w), comp.dynamic_quality)
        t1 = time.perf_counter()
        modes = _leaf_modes(comp.root, Counter())
        root, _, q = codec.decompress(bitstream)
        comp.orig_h, comp.orig_w = h, w
        rec = comp.reconstruct(root, override_quality=q)
        t2 = time.perf_counter()
        rows.append({**base, "param": lam, "bpp": len(bitstream) * 8 / (h * w), "enc_time": t1 - t0,
                     "dec_time": t2 - t1, "leaves_flat": modes["flat"], "leaves_interp": modes["interp"],
                     "leaves_dct": modes["dct"], **_metrics(img, rec, ref_map)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="results/v2/params.json")
    ap.add_argument("--out", default="results/v2/benchmark.csv")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    params = json.loads(Path(args.params).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with out.open() as f:
            done = {(r["dataset"], r["image"], r["method"]) for r in csv.DictReader(f)}
    methods = ["base"] + SALIENCY_NETS + ["jpeg", "webp"]
    tasks = [(d, str(p), m) for d in args.datasets for p in list_images(d) for m in methods
             if (d, p.name, m) not in done]
    print(f"{len(tasks)} tareas pendientes ({len(done)} ya hechas). Params: {params}", flush=True)

    new_file = not out.exists()
    t0 = time.time()
    with out.open("a", newline="") as f, Pool(args.workers, initializer=_init, initargs=(params,)) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, rows in enumerate(pool.imap_unordered(run_task, tasks), 1):
            writer.writerows(rows)
            f.flush()
            if i % 20 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"[{i}/{len(tasks)}] {el / 60:.1f} min, ETA {el / i * (len(tasks) - i) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
