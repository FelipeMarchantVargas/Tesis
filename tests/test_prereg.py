"""Componentes de las pruebas pre-registradas (prereg_tests.py)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiment_fixations import pair_bd
from prereg_tests import bootstrap_median_ci, holm, valid


def test_holm_matches_hand_computation():
    # ordenados: 0,01*3 = 0,03; 0,03*2 = 0,06; 0,04*1 = 0,04 -> monotonía: 0,06
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_bootstrap_ci_contains_median_and_is_reproducible():
    x = np.random.default_rng(0).normal(-3, 2, 60)
    lo, hi = bootstrap_median_ci(x)
    assert lo < np.median(x) < hi
    assert (lo, hi) == bootstrap_median_ci(x)


def _grid():
    rows = []
    for arm, scale in [("base", 1.0), ("better", 0.8)]:  # 'better': 20 % menos bits a igual calidad
        for k in range(1, 8):
            rows.append({"image": "im", "arm": arm, "eta": 0.0 if arm == "base" else 2.0,
                         "bpp": scale * 0.2 * k, "fw_lpips_full": 0.5 / k})
    return pd.DataFrame(rows)


def test_pair_bd_self_is_zero_and_detects_known_saving():
    g = _grid()
    bd, ov = pair_bd(g, ("base", 0.0), ("base", 0.0), "fw_lpips_full", True)["im"]
    assert abs(bd) < 1e-9 and ov == 1.0
    bd, ov = pair_bd(g, ("base", 0.0), ("better", 2.0), "fw_lpips_full", True)["im"]
    assert abs(bd - (-20.0)) < 1e-6 and ov == 1.0


def test_valid_drops_low_overlap_and_nan():
    assert valid({"a": (-5.0, 0.9), "b": (-5.0, 0.3), "c": (np.nan, 1.0)}) == {"a": -5.0}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("OK")
