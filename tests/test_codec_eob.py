"""Regresión: un bloque DCT con el coeficiente 63 no nulo no debe desincronizar el bitstream.

Antes del fix, _read_dct_block salía del bucle al llegar idx a 64 sin consumir el
marcador EOB, y todas las hojas siguientes se decodificaban desfasadas.
Ejecutar: python -m pytest tests/  (o python tests/test_codec_eob.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.codec import QuadtreeCodec
from src.quadtree import QuadtreeNode


def _leaves(node, acc):
    if node.children:
        for child in node.children:
            _leaves(child, acc)
    else:
        acc.append(node)
    return acc


def test_dct_block_with_last_coefficient_roundtrip():
    root = QuadtreeNode(0, 0, 16, 16, 0)
    root.children = [QuadtreeNode(y, x, 8, 8, 1) for y, x in [(0, 0), (0, 8), (8, 0), (8, 8)]]

    coeffs = np.zeros(64, dtype=np.int16)
    coeffs[0], coeffs[5], coeffs[63] = 40, -3, 7  # coeficiente 63 no nulo
    corners = [np.array([200, 100, 50], np.float32)] * 4
    dct_leaf = root.children[0]
    dct_leaf.mode, dct_leaf.dct_coeffs = 'dct', coeffs
    dct_leaf.color_tl = dct_leaf.color_tr = dct_leaf.color_bl = dct_leaf.color_br = corners[0]

    for i, leaf in enumerate(root.children[1:]):
        leaf.mode = 'flat'
        leaf.color_tl = np.array([10 + 50 * i, 120, 230 - 50 * i], np.float32)

    codec = QuadtreeCodec()
    decoded, _, _ = codec.decompress(codec.compress(root, (16, 16), 90))

    out = _leaves(decoded, [])
    assert [n.mode for n in out] == ['dct', 'flat', 'flat', 'flat']
    np.testing.assert_array_equal(out[0].dct_coeffs, coeffs)
    for original, leaf in zip(root.children[1:], out[1:]):
        np.testing.assert_allclose(leaf.color_tl, original.color_tl, atol=2.0)


if __name__ == "__main__":
    test_dct_block_with_last_coefficient_roundtrip()
    print("OK")
