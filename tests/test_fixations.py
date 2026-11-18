"""Pruebas de sanidad del protocolo con fijaciones reales (detección, densidad G, métricas FW-*, qmap)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fixations import detect_fixations, fixation_density, rank_normalize_cells
from src.metrics import background_weights, fw_lpips, fw_mse, fw_psnr, lpips_spatial_map
from src.quadtree import QuadtreeCompressor


def _trace(points, dwell=240, travel=8, noise=1.0, seed=0):
    """Registro sintético a 240 Hz: fijaciones de `dwell` muestras unidas por sacadas lineales."""
    rng = np.random.default_rng(seed)
    segs = []
    for i, p in enumerate(points):
        segs.append(np.tile(p, (dwell, 1)) + rng.normal(0, noise, (dwell, 2)))
        if i + 1 < len(points):
            t = np.linspace(0, 1, travel + 2)[1:-1, None]
            segs.append(np.asarray(p) + t * (np.asarray(points[i + 1]) - p))
    return np.vstack(segs)


def test_detects_each_fixation_and_its_location():
    pts = [(200, 300), (700, 400), (500, 150)]
    fix = detect_fixations(_trace(pts))
    assert len(fix) == 3
    np.testing.assert_allclose(fix[:, :2], pts, atol=2)
    assert np.all(np.diff(fix[:, 2]) > 0)  # orden temporal


def test_steady_gaze_is_single_fixation():
    fix = detect_fixations(_trace([(512, 384)], dwell=720))
    assert len(fix) == 1
    np.testing.assert_allclose(fix[0, :2], (512, 384), atol=1)


def test_density_is_normalized_and_peaks_at_point():
    G = fixation_density([(30, 20), (500, 500), (np.nan, 3)], (64, 80), sigma=4)
    assert G.shape == (64, 80) and np.isclose(G.sum(), 1.0)
    assert np.unravel_index(G.argmax(), G.shape) == (20, 30)
    U = fixation_density(np.empty((0, 2)), (8, 8))
    np.testing.assert_allclose(U, 1 / 64)


def test_fw_mse_uniform_equals_mse_and_ignores_unweighted_error():
    rng = np.random.default_rng(1)
    a = (rng.random((40, 50, 3)) * 255).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-20, 20, a.shape), 0, 255).astype(np.uint8)
    U = np.full((40, 50), 1 / 2000)
    assert np.isclose(fw_mse(a, b, U), np.mean((a.astype(float) - b) ** 2))
    G = fixation_density([(5, 5)], (40, 50), sigma=0)  # masa puntual en (5, 5)
    c = a.copy()
    c[20:, 25:] = 0  # error solo lejos del punto
    assert fw_mse(a, c, G) == 0 and fw_psnr(a, c, G) == float("inf")


def test_background_weights_normalized_and_zero_at_peak():
    G = fixation_density([(10, 10)], (32, 32), sigma=3)
    B = background_weights(G)
    assert np.isclose(B.sum(), 1.0) and B[10, 10] == 0


def test_fw_lpips_uniform_matches_global_lpips():
    import lpips
    import torch
    rng = np.random.default_rng(2)
    a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-40, 40, a.shape), 0, 255).astype(np.uint8)
    spatial = lpips.LPIPS(net="alex", spatial=True, verbose=False).eval()
    glob = lpips.LPIPS(net="alex", verbose=False).eval()
    m = lpips_spatial_map(spatial, a, b)
    t = lambda x: torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1)  # noqa: E731
    with torch.no_grad():
        g = glob(t(a), t(b)).item()
    assert m.shape == (64, 64)
    assert np.isclose(fw_lpips(m, np.full((64, 64), 1 / 64 ** 2)), g, rtol=0.05)


def test_rank_normalized_guide_gives_balanced_qmap():
    for shape in [(96, 128), (100, 130)]:
        _check_balanced_qmap(shape)


def _check_balanced_qmap(shape):
    h, w = shape
    rng = np.random.default_rng(3)
    smap = fixation_density(rng.random((5, 2)) * (w, h), shape, sigma=6).astype(np.float32)
    guide = rank_normalize_cells(smap, cell=32)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    comp = QuadtreeCompressor(8, 10)
    comp.compress(img, guide, 0.0, 0.0, lam=10, beta=0.0, quality=50, qmap_strength=2.0)
    ny, nx = -(-h // 32), -(-w // 32)
    levels = comp.qmap[:ny, :nx].ravel()
    counts = np.bincount(levels, minlength=4)
    assert counts.max() - counts.min() <= 1  # misma proporción de celdas por nivel
    # el orden de las celdas sigue al mapa original
    cell_mean = np.array([[smap[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32].mean() for j in range(nx)]
                          for i in range(ny)]).ravel()
    assert levels[np.argmax(cell_mean)] == 3 and levels[np.argmin(cell_mean)] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("OK")
