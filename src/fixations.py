"""Fijaciones humanas (MIT1003) para evaluar sin circularidad.

- detect_fixations: port de checkFixations.m (Hidalgo-Sotelo 2006, DatabaseCode de MIT1003),
  algoritmo de aceleración 'A'. Índices 1-based internamente para seguir el original línea a línea.
- fixation_density: mapa G (suma 1) a partir de puntos de fijación.
- rank_normalize_cells: normaliza un mapa guía por rangos a nivel de celda qmap, para que todos los
  brazos tengan la misma proporción de celdas en cada nivel y solo difieran en *dónde*.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# Parámetros de checkFixations.m (display 1024x768 px = 40,5x30 cm a 75 cm)
PX_PER_DEG = 34
SAMPLING_RATE = 240
_WIN, _STEP = 4, 1
_ACCEL_CRIT = PX_PER_DEG * 6
_FIXDUR_PTS = round(50 * SAMPLING_RATE / 1000)


def _acceleration(data):
    n = len(data)
    pos = np.concatenate([[0.0], np.hypot(np.diff(data[:, 0]), np.diff(data[:, 1]))])
    sm = np.convolve(pos, np.ones(_WIN) / _WIN)[:n]  # filter(ones/winSize, 1, x)
    c = np.concatenate([[0.0], np.cumsum(sm)])  # sum(sm(a:b)) 1-based = c[b] - c[a-1]
    k = np.arange(2, n - _WIN - _STEP + 1)       # k = 2 : n-winSize-stepSize (1-based)
    accel = np.zeros(n)
    v1 = c[k + _WIN] - c[k - 1]                   # sum(sm(k : k+winSize))
    v2 = c[k + _WIN + _STEP] - c[k + _STEP - 1]   # sum(sm(k+stepSize : k+winSize+stepSize))
    accel[k - 1] = (v2 - v1) / (_WIN / SAMPLING_RATE)
    return accel


def _saccade_labels(data, accel):
    """Etiquetas por muestra (0 fijación, 3 inicio, 1 medio, 2 fin de sacada) o None si no hay sacadas."""
    n = len(data)
    A = lambda i: accel[np.asarray(i) - 1]  # noqa: E731  (acceso 1-based)
    check = np.flatnonzero(accel > _ACCEL_CRIT) + 1
    if len(check) < 3:
        return None
    d = np.diff(check)
    e1 = list(np.flatnonzero(d == 1) + 1)
    e2 = list(np.flatnonzero(d > 1) + 1)
    e3 = [e2[i + 1] for i in np.flatnonzero(np.diff(e2) == 1)] if len(e2) > 1 else []
    if check[-1] - check[-2] == 1:
        e1.append(len(check))
    else:
        e3.append(len(check))
    m = np.zeros((len(check), 2), dtype=np.int64)
    m[:, 0] = check
    for idx, lab in ((e1, 1), (e2, 2), (e3, 3)):
        if idx:
            m[np.asarray(idx) - 1, 1] = lab
    if check[-1] - check[-2] == 1:
        m[-1, 1] = 2
    else:
        m, e2 = m[:-1], e2[:-1]
    if e2:
        m[np.asarray(e2), 1] = 3  # eachsac3 = eachsac2 + 1 (1-based) -> índice 0-based eachsac2
    if m[0, 1] == 1:
        m[0, 1] = 3

    kept = []
    if len(m) >= 2 and m[0, 1] == 3 and m[1, 1] == 1:
        kept.append(m[0])
    for i in range(1, len(m) - 1):  # i = 2 : length(n)-1 (1-based)
        lab = m[i, 1]
        if (lab == 3 and m[i + 1, 1] == 1) or lab == 1 or (lab == 2 and m[i - 1, 1] == 1 and m[i + 1, 1] == 3):
            kept.append(m[i])
        if i == len(m) - 2:
            last = kept[-1][0] if kept else 0
            if m[-1, 0] - last == 1:
                kept.append(m[-1])
    if len(kept) < 3:
        return None
    cs = np.array(kept)

    ends = np.flatnonzero(cs[:, 1] == 2)
    starts = np.flatnonzero(cs[:, 1] == 3)
    good, blah = [], []
    last_nonzero = np.flatnonzero(accel != 0)[-1] + 1 if np.any(accel != 0) else 0
    for s, row in enumerate(ends):
        chk = np.array([cs[row, 0] + 1, cs[row, 0] + 2])
        cross = A(chk) < -_ACCEL_CRIT
        stop, ok = 25, True
        while not (cross[0] and not cross[1]) and ok:
            chk = chk + 1
            cross = A(chk) < -_ACCEL_CRIT
            stop -= 1
            ok = chk.max() < n and stop > 0
        if not ok:
            if np.all(cs[ends, 0] == last_nonzero):  # el ensayo termina en sacada
                good.append(s)
                blah.append(len(accel))
        else:
            good.append(s)
            blah.append(chk[0])
    good = [s for s in good if s < len(starts)]  # MATLAB fallaría si hay menos inicios que fines
    if not good:
        return np.zeros(n, dtype=np.int64)  # se descartaron todas: sin sacadas etiquetadas
    g2, g3 = ends[good], starts[good]
    decel = np.column_stack([cs[g2, 0], blah])
    if len(blah) > 1:
        too_short = np.flatnonzero(cs[g3[1:], 0] - decel[:-1, 1] <= _FIXDUR_PTS)
        start_sac = cs[g3[np.setdiff1d(np.arange(len(blah)), too_short + 1)], 0]
        end_sac = decel[np.setdiff1d(np.arange(len(blah)), too_short)].copy()
    else:
        start_sac = cs[g3, 0]
        end_sac = decel[-1:].copy()
    if n - decel[-1, 1] <= _FIXDUR_PTS:
        end_sac[-1, 1] = n
    lab = np.zeros(n, dtype=np.int64)
    lab[start_sac - 1] = 3
    lab[end_sac[:, 1] - 1] = 2
    for a, b in zip(start_sac, end_sac[:, 1]):
        lab[a:b - 1] = 1  # (a+1):(b-1) en 1-based
    return lab


def detect_fixations(eye_data):
    """Fijaciones de un registro (N x 2, px de pantalla; negativos = parpadeos), como checkFixations.m.

    Devuelve un arreglo (k, 4): x, y (medianas), inicio, fin (muestras 1-based), en orden temporal.
    """
    data = np.asarray(eye_data, dtype=np.float64)[:, :2]
    n = len(data)

    def one_fixation():
        return np.array([[np.nanmedian(data[:, 0]), np.nanmedian(data[:, 1]), 1, n]])

    if n < _WIN + _STEP + 3:
        return one_fixation()
    lab = _saccade_labels(data, _acceleration(data))
    if lab is None:
        return one_fixation()
    e3 = np.flatnonzero(lab == 3) + 1
    e2 = np.flatnonzero(lab == 2) + 1
    fixs = np.flatnonzero(lab == 0) + 1
    if n - len(fixs) < 3 or len(e3) == 0 or len(e2) == 0:
        return one_fixation()
    if len(e3) == 1:
        start = [fixs[0]] + (list(e2 + 1) if e2[-1] != n else [])
    else:
        start = [fixs[0]] + list(e2[:-1] + 1) + ([e2[-1] + 1] if e2[-1] != n else [])
    c1, c2 = e3[0] == 1, e2[-1] == n
    end = list(e3[1:] - 1 if c1 else e3 - 1) + ([] if c2 else [n])
    out = []
    for a, b in zip(start, end):
        seg = data[a - 1:b]
        out.append([np.nanmedian(seg[:, 0]), np.nanmedian(seg[:, 1]), a, b])
    return np.array(out)


def fixation_density(points, shape, sigma=PX_PER_DEG):
    """Mapa de densidad G (h, w), suma 1: impulsos en los puntos (x, y) suavizados con una gaussiana.

    sigma por defecto = 1° de ángulo visual en el montaje de MIT1003. Puntos fuera de la imagen se
    descartan; sin puntos válidos devuelve G uniforme.
    """
    h, w = shape
    acc = np.zeros((h, w), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    xi, yi = np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    np.add.at(acc, (yi[ok], xi[ok]), 1.0)
    if acc.sum() == 0:
        return np.full((h, w), 1.0 / (h * w))
    g = gaussian_filter(acc, sigma, mode="constant") if sigma > 0 else acc
    return g / g.sum()


def padded_size(h, w):
    """Lado del cuadrado al que QuadtreeCompressor rellena la imagen (src/quadtree.py)."""
    return max(1 << (max(h, w) - 1).bit_length(), 16)


def rank_normalize_cells(smap, cell=32, seed=0):
    """Reemplaza cada celda qmap por su rango (en (0, 1)) entre las celdas que tocan la imagen.

    QuadtreeCompressor rellena con ceros hasta un cuadrado de lado potencia de 2 y toma la media de
    cada celda; para una celda parcial el valor se divide por su fracción válida, de modo que su
    media tras el relleno sea exactamente el rango (puede superar 1 en los píxeles; con beta = 0
    solo lo usa el qmap). Los empates (p. ej. fondo con densidad ~0) se rompen al azar con `seed`,
    para que la proporción de celdas por nivel sea exactamente la misma en todos los brazos.
    """
    h, w = smap.shape
    n = padded_size(h, w) // cell
    pad = np.zeros((n * cell, n * cell), dtype=np.float64)
    valid = np.zeros_like(pad)
    pad[:h, :w], valid[:h, :w] = smap, 1.0
    s = pad.reshape(n, cell, n, cell).sum(axis=(1, 3))
    frac = valid.reshape(n, cell, n, cell).mean(axis=(1, 3))
    inside = frac > 0
    mean = np.where(inside, s / np.maximum(frac * cell * cell, 1), 0.0)
    r = np.zeros_like(mean)
    vals = mean[inside]
    order = np.lexsort((np.random.default_rng(seed).random(vals.size), vals))
    ranks = np.empty(vals.size)
    ranks[order] = np.arange(1, vals.size + 1)
    r[inside] = (ranks - 0.5) / vals.size
    per_cell = np.where(inside, r / np.maximum(frac, 1e-12), 0.0)
    out = np.kron(per_cell, np.ones((cell, cell)))[:h, :w]
    return out.astype(np.float32)
