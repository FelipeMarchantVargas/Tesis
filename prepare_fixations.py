"""Prepara MIT1003 para el protocolo con fijaciones reales (fase 1).

Entradas (data/mit1003/, de http://people.csail.mit.edu/tjudd/WherePeopleLook/):
  ALLSTIMULI/*.jpeg, DATA/<usuario>/<imagen>.mat (eyeData crudo a 240 Hz), ALLFIXATIONMAPS/*_fixMap.jpg
Salidas (data/mit1003/derived/):
  fixations.csv    image, user, k, x, y, start, end, half  (todas las fijaciones, k = orden temporal)
  split.csv        image, h, w, split (calib/test, semilla 42)
  center_bias.npy  promedio de G sobre la partición de calibración (256x256, coords. normalizadas)
  config.json      convención elegida (desplazamiento de coordenadas, primera fijación)
Validación (results/v2/fixations/validation.csv): correlación de Pearson (CC) entre nuestro G y el
mapa oficial, para cada convención; se elige la de mayor CC mediana. sigma queda fijo en 1° (34 px).
"""
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.fixations import PX_PER_DEG, detect_fixations, fixation_density

ROOT = Path("data/mit1003")
OUT = ROOT / "derived"
VAL = Path("results/v2/fixations")
SCREEN_W, SCREEN_H = 1024, 768  # checkFixations.m: display 1024x768 px
N_CALIB, SEED, N_VALIDATE = 200, 42, 100
CB_SIZE = 256


def _load_eye(mat_path):
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    rec = next(v for k, v in m.items() if not k.startswith("__"))
    data = rec.DATA[0] if np.ndim(rec.DATA) else rec.DATA
    name = getattr(rec, "imgName", mat_path.stem + ".jpeg")
    return str(name), np.asarray(data.eyeData, dtype=np.float64)


def _user_fixations(user_dir):
    rows = []
    for mat in sorted(user_dir.glob("*.mat")):
        try:
            _, eye = _load_eye(mat)
        except Exception as e:  # noqa: BLE001
            print(f"  {user_dir.name}/{mat.name}: {e}")
            continue
        if eye.ndim != 2 or len(eye) == 0:
            continue
        for k, (x, y, a, b) in enumerate(detect_fixations(eye), start=1):
            # La imagen se identifica por el nombre del .mat, como en el README de MIT1003: en 4 archivos
            # el campo imgName apunta a otra imagen y los datos coinciden con el nombre del archivo
            # (CC 0,90-0,94 contra el mapa oficial, frente a 0,07-0,37 con imgName).
            rows.append((mat.stem, user_dir.name, k, x, y, int(a), int(b)))
    return rows


def points(fx, w, h, offset, skip_first):
    """Puntos (x, y) en coordenadas de imagen según la convención."""
    sel = fx[fx.k > 1] if skip_first else fx
    p = sel[["x", "y"]].to_numpy(dtype=np.float64)
    if offset == "centered":  # imagen centrada en la pantalla de 1024x768
        p = p - [(SCREEN_W - w) / 2, (SCREEN_H - h) / 2]
    return p


def _cc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _validate(task):
    stem, w, h, fx = task
    ref = cv2.imread(str(ROOT / "ALLFIXATIONMAPS" / f"{stem}_fixMap.jpg"), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        return []
    ref = ref.astype(np.float64)
    if ref.shape != (h, w):
        ref = cv2.resize(ref, (w, h))
    out = []
    for offset in ("raw", "centered"):
        for skip in (False, True):
            G = fixation_density(points(fx, w, h, offset, skip), (h, w), sigma=PX_PER_DEG)
            out.append({"image": stem, "offset": offset, "skip_first": skip, "cc": _cc(G, ref)})
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    VAL.mkdir(parents=True, exist_ok=True)

    sizes = {}  # el mapa oficial tiene el tamaño de la imagen; así no hace falta ALLSTIMULI aquí
    for p in sorted((ROOT / "ALLFIXATIONMAPS").glob("*_fixMap.jpg")):
        sizes[p.name[: -len("_fixMap.jpg")]] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape[:2]
    print(f"{len(sizes)} imágenes")

    users = sorted(d for d in (ROOT / "DATA").iterdir() if d.is_dir())
    with Pool() as pool:
        rows = [r for part in pool.map(_user_fixations, users) for r in part]
    fix = pd.DataFrame(rows, columns=["image", "user", "k", "x", "y", "start", "end"])
    fix = fix[fix.image.isin(sizes)]
    print(f"{len(users)} usuarios, {len(fix)} fijaciones, {fix.image.nunique()} imágenes con datos")

    # Validación de la convención de coordenadas contra los mapas oficiales
    rng = np.random.default_rng(SEED)
    sample = rng.choice(sorted(fix.image.unique()), size=min(N_VALIDATE, fix.image.nunique()), replace=False)
    groups = dict(tuple(fix[fix.image.isin(sample)].groupby("image")))
    tasks = [(s, sizes[s][1], sizes[s][0], groups[s]) for s in sample]
    with Pool() as pool:
        val = pd.DataFrame([r for part in pool.map(_validate, tasks) for r in part])
    val.to_csv(VAL / "validation.csv", index=False)
    summary = val.groupby(["offset", "skip_first"]).cc.agg(["median", "mean", "min"]).round(3)
    print(summary.to_string())
    best = val.groupby(["offset", "skip_first"]).cc.median().idxmax()
    config = {"offset": best[0], "skip_first": bool(best[1]), "sigma_px": PX_PER_DEG,
              "cc_median": float(summary.loc[best, "median"])}
    print("convención:", config)

    # Partición y mitades de observadores para el oráculo cruzado
    stems = sorted(fix.image.unique())
    perm = np.random.default_rng(SEED).permutation(len(stems))
    split = pd.DataFrame({"image": stems, "h": [sizes[s][0] for s in stems], "w": [sizes[s][1] for s in stems]})
    split["split"] = "test"
    split.loc[perm[:N_CALIB], "split"] = "calib"
    split.to_csv(OUT / "split.csv", index=False)

    half = {}
    for i, s in enumerate(stems):
        us = sorted(fix.loc[fix.image == s, "user"].unique())
        order = np.random.default_rng(SEED + i).permutation(len(us))
        for j, idx in enumerate(order):
            half[(s, us[idx])] = "A" if j < len(us) // 2 else "B"
    fix["half"] = [half[(s, u)] for s, u in zip(fix.image, fix.user)]
    fix.to_csv(OUT / "fixations.csv", index=False)

    # Sesgo central empírico (solo calibración)
    acc = np.zeros((CB_SIZE, CB_SIZE))
    calib = split[split.split == "calib"]
    for s, h, w in calib[["image", "h", "w"]].itertuples(index=False):
        G = fixation_density(points(fix[fix.image == s], w, h, config["offset"], config["skip_first"]),
                             (h, w), sigma=PX_PER_DEG)
        acc += cv2.resize(G, (CB_SIZE, CB_SIZE), interpolation=cv2.INTER_AREA)
    np.save(OUT / "center_bias.npy", acc / acc.sum())
    config.update(n_images=len(stems), n_calib=int(len(calib)), n_users=len(users), n_fixations=len(fix))
    (OUT / "config.json").write_text(json.dumps(config, indent=2))
    (VAL / "config.json").write_text(json.dumps(config, indent=2))
    print("listo:", OUT)
