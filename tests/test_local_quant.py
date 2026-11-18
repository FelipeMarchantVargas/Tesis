"""El mapa de cuantización local (versión 4 del bitstream) debe reconstruirse igual en el decoder."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.codec import QuadtreeCodec
from src.quadtree import QuadtreeCompressor


def _roundtrip(eta):
    rng = np.random.default_rng(0)
    img = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
    img[:, :64] = 128  # mitad plana, mitad ruidosa
    smap = np.zeros(img.shape[:2], np.float32)
    smap[:, 64:] = 1.0
    comp = QuadtreeCompressor(8, 10)
    comp.compress(img, smap, 0.0, 0.0, lam=10, beta=0.0, quality=50, qmap_strength=eta)
    encoder_rec = comp.reconstruct(comp.root)
    codec = QuadtreeCodec()
    bs = codec.compress(comp.root, (comp.root.h, comp.root.w), comp.dynamic_quality,
                        qmap=comp.qmap, qmap_strength=comp.qmap_strength)
    root, _, q = codec.decompress(bs)
    dec = QuadtreeCompressor(8, 10)
    dec.orig_h, dec.orig_w = img.shape[:2]
    dec.qmap, dec.qmap_strength = codec.qmap, codec.qmap_strength
    dec.qmap_cell, dec.qmap_levels = codec.qmap_cell, codec.qmap_levels
    return encoder_rec, dec.reconstruct(root, override_quality=q), comp, len(bs)


def test_local_quant_roundtrip_matches_encoder():
    enc, dec, comp, _ = _roundtrip(2.0)
    assert comp.qmap is not None and comp.qmap.max() > comp.qmap.min()
    np.testing.assert_allclose(enc.astype(float), dec.astype(float), atol=2)


def test_eta_zero_is_original_codec():
    _, _, comp, _ = _roundtrip(0.0)
    assert comp.qmap is None


if __name__ == "__main__":
    test_local_quant_roundtrip_matches_encoder()
    test_eta_zero_is_original_codec()
    print("OK")
