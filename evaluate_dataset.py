import os
import cv2
import pandas as pd
import time
from pathlib import Path
from src.metrics import QualityMetrics
from src.quadtree import QuadtreeCompressor
from src.saliency import SaliencyDetector
import numpy as np
import pickle
import zlib

def evaluate():
    dataset_path = Path("data/kodak")
    output_csv = "results/benchmark_results_full.csv" # Nombre nuevo para diferenciar
    model_path = "models/u2net.pth"
    
    # === CONFIGURACIÓN DE COMPETIDORES ===
    
    # 1. TU MÉTODO (Optimizado)
    # Rango amplio para cubrir desde muy bajo bitrate hasta calidad media
    qt_thresholds = [10, 25, 40, 60, 90, 150, 250] 
    alpha = 0.17 # Tu valor mágico de Optuna

    # 2. JPEG (El Clásico)
    jpeg_qualities = [5, 10, 20, 40, 70]

    # 3. WebP (El Moderno - Tu nuevo rival)
    webp_qualities = [5, 10, 20, 40, 70]

    print("--- Iniciando Benchmark Completo (Ours vs JPEG vs WebP) ---")
    
    metrics_engine = QualityMetrics()
    saliency_detector = SaliencyDetector(weights_path=model_path)
    
    results = []
    images = sorted(list(dataset_path.glob("*.png")))
    
    if not images:
        print("ERROR: Faltan imágenes en data/kodak")
        return

    for img_path in images:
        print(f"\nProcesando {img_path.name}...")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        pixels = h * w

        # Generar Saliencia
        saliency_map = saliency_detector.get_saliency_map(img_rgb)

        # --- A. TU MÉTODO ---
        for th in qt_thresholds:
            compressor = QuadtreeCompressor(min_block_size=4, max_depth=10)
            compressor.compress(img_rgb, saliency_map, threshold=th, alpha=alpha)
            rec_rgb = compressor.reconstruct((h, w))
            
            # Peso Real
            payload = {'leaves': compressor.leaves, 'shape': (h, w)}
            compressed = zlib.compress(pickle.dumps(payload), level=9)
            bpp = (len(compressed) * 8) / pixels
            
            scores = metrics_engine.calculate_all(img_rgb, rec_rgb)
            results.append({"Image": img_path.name, "Method": "Ours", "Param": th, "BPP": bpp, **scores})
            print(f"  [Ours Th={th}] BPP: {bpp:.3f} | LPIPS: {scores['lpips']:.3f} | VIF: {scores['vif']:.3f}")

        # --- B. JPEG ---
        for q in jpeg_qualities:
            temp = "temp.jpg"
            cv2.imwrite(temp, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            
            bpp = (os.path.getsize(temp) * 8) / pixels
            rec = cv2.cvtColor(cv2.imread(temp), cv2.COLOR_BGR2RGB)
            
            scores = metrics_engine.calculate_all(img_rgb, rec)
            results.append({"Image": img_path.name, "Method": "JPEG", "Param": q, "BPP": bpp, **scores})
            # print(f"  [JPEG Q={q}] BPP: {bpp:.3f}")

        # --- C. WebP ---
        for q in webp_qualities:
            temp = "temp.webp"
            cv2.imwrite(temp, img_bgr, [cv2.IMWRITE_WEBP_QUALITY, q])
            
            bpp = (os.path.getsize(temp) * 8) / pixels
            rec = cv2.cvtColor(cv2.imread(temp), cv2.COLOR_BGR2RGB)
            
            scores = metrics_engine.calculate_all(img_rgb, rec)
            results.append({"Image": img_path.name, "Method": "WebP", "Param": q, "BPP": bpp, **scores})
            print(f"  [WebP Q={q}] BPP: {bpp:.3f} | LPIPS: {scores['lpips']:.3f}")

    # Guardar
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nResultados guardados en {output_csv}")
    if os.path.exists("temp.jpg"): os.remove("temp.jpg")
    if os.path.exists("temp.webp"): os.remove("temp.webp")

if __name__ == "__main__":
    evaluate()