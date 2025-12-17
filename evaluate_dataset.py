import os
import cv2
import pandas as pd
import time
from pathlib import Path
import numpy as np

# Importaciones de tu proyecto
from src.metrics import QualityMetrics
from src.quadtree import QuadtreeCompressor
from src.saliency import SaliencyDetector
from src.codec import QuadtreeCodec

def evaluate():
    # --- CONFIGURACIÓN ---
    dataset_path = Path("data/kodak")
    output_csv = "results/benchmark_results_final.csv" 
    model_path = "models/u2net.pth"
    
    FIXED_LOW_TH = 19.9523
    ALPHA_OPT    = 7.7880
    BETA_OPT     = 4.6632
    
    # Barrido de Lambdas (Control de Calidad vs Peso)
    rdo_lambdas = [1, 5, 10, 20, 40, 70, 110, 150]
    
    # Calidades de referencia
    jpeg_qualities = [10, 20, 40, 60, 80, 95]
    webp_qualities = [10, 20, 40, 60, 80, 95]

    # Inicializar Modelos
    saliency_detector = SaliencyDetector(weights_path=model_path)
    metrics = QualityMetrics()
    codec = QuadtreeCodec()
    
    if not dataset_path.exists():
        print(f"Error: No se encuentra {dataset_path}")
        return
        
    images = sorted(list(dataset_path.glob("*.png")))
    results = []

    print(f"Iniciando evaluación FINAL con Parámetros Optimizados...")
    print(f"Configuración: Th={FIXED_LOW_TH:.2f}, Alpha={ALPHA_OPT:.2f}, Beta={BETA_OPT:.2f}")

    for img_path in images:
        print(f"\nProcesando: {img_path.name}")
        
        # Cargar Imagen
        img_rgb = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Mapa de Saliencia
        smap = saliency_detector.get_saliency_map(img_rgb)
        
        # --- 1. OUR METHOD (Multi-Mode RDO + Saliency) ---
        compressor = QuadtreeCompressor(min_block_size=4, max_depth=12)
        
        for lam in rdo_lambdas:
            start_time = time.time()
            
            # Compresión: 
            # - Threshold bajo (5) para crear un árbol detallado.
            # - RDO podará usando Lambda y protegerá con Beta=2.0 (Saliencia).
            # - Ahora el RDO elige automáticamente entre 'Flat' y 'Interp'.
            compressor.compress(img_rgb, smap, threshold=FIXED_LOW_TH, alpha=ALPHA_OPT, lam=lam, beta=BETA_OPT)
            
            # Codificación (El codec detecta automáticamente los modos guardados en los nodos)
            compressed_bytes = codec.compress(compressor.root, (h, w), mode='optimized')
            bpp = len(compressed_bytes) * 8 / (h * w)
            t_enc = time.time() - start_time
            
            # Reconstrucción
            rec_root, _ = codec.decompress(compressed_bytes)
            rec_img = compressor.reconstruct(rec_root) # El reconstruct lee el modo (flat/interp) del nodo
            
            m = metrics.calculate_all(img_rgb, rec_img, saliency_map=smap)
            
            # Guardamos con el nombre nuevo
            res = {
                "Image": img_path.name,
                "Method": "Ours (Multi-Mode RDO)", 
                "Param": lam,
                "BPP": bpp,
                "Time_s": t_enc,
                **m
            }
            results.append(res)
            print(f"  [Ours] Lam={lam}: BPP={bpp:.3f}, SSIM={m['ssim']:.3f}, SW-SSIM={m['sw_ssim']:.3f}")

        # --- 2. BASELINE (Standard RDO - Sin Saliencia) ---
        for lam in rdo_lambdas:
            start_time = time.time()
            
            # Beta=0.0 apaga la protección de saliencia. Es un RDO matemático puro.
            compressor.compress(img_rgb, smap, threshold=FIXED_LOW_TH, alpha=0.0, lam=lam, beta=0.0)
            
            compressed_bytes = codec.compress(compressor.root, (h, w), mode='optimized')
            bpp = len(compressed_bytes) * 8 / (h * w)
            t_enc = time.time() - start_time
            
            rec_root, _ = codec.decompress(compressed_bytes)
            rec_img = compressor.reconstruct(rec_root)
            
            m = metrics.calculate_all(img_rgb, rec_img, saliency_map=smap)
            
            res = {
                "Image": img_path.name,
                "Method": "Standard RDO (No Saliency)",
                "Param": lam,
                "BPP": bpp,
                "Time_s": t_enc,
                **m
            }
            results.append(res)

        # --- 3. JPEG & WebP (Referencias) ---
        # (Este bloque se mantiene igual para tener contexto)
        for q in jpeg_qualities:
            _, enc = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
            bpp = len(enc) * 8 / (h * w)
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            m = metrics.calculate_all(img_rgb, dec)
            results.append({"Image": img_path.name, "Method": "JPEG", "Param": q, "BPP": bpp, **m})

        for q in webp_qualities:
            _, enc = cv2.imencode('.webp', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, q])
            bpp = len(enc) * 8 / (h * w)
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            m = metrics.calculate_all(img_rgb, dec)
            results.append({"Image": img_path.name, "Method": "WebP", "Param": q, "BPP": bpp, **m})

    # Guardar CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nEvaluación completa. Resultados guardados en {output_csv}")

if __name__ == "__main__":
    evaluate()