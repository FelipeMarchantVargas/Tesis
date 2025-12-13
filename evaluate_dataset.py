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
    
    # 1. Parámetros Científicos (Optimizados con Bayes/Optuna)
    # Alpha controla qué tanto caso le hacemos a la CNN.
    # El valor 4.7557 indica una fuerte dependencia de la atención visual.
    ALPHA_OPT = 4.7557 
    
    # 2. Barrido de Thresholds (Curva Rate-Distortion)
    # Empezamos en tu óptimo (10) y subimos para generar puntos de mayor compresión.
    # Escala progresiva para cubrir varios rangos de bitrate.
    qt_thresholds = [10, 15, 20, 30, 45, 70, 110, 150] 
    
    # Calidades de referencia estándar
    jpeg_qualities = [5, 10, 20, 30, 50, 70, 80, 90]
    webp_qualities = [5, 10, 20, 30, 50, 70, 80, 90]

    print(f"--- Iniciando Benchmark Final ---")
    print(f"Alpha Óptimo: {ALPHA_OPT}")
    print(f"Thresholds: {qt_thresholds}")
    
    # Inicializar Motores
    metrics_engine = QualityMetrics()
    saliency_detector = SaliencyDetector(weights_path=model_path)
    codec = QuadtreeCodec()
    
    results = []
    images = sorted(list(dataset_path.glob("*.png")))
    
    if not images:
        print("Error: No se encontraron imágenes en data/kodak")
        return

    os.makedirs("results", exist_ok=True)

    for img_path in images:
        print(f"\nProcesando {img_path.name}...")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        pixels = h * w
        
        # Mapa de Saliencia (Se calcula una sola vez por imagen)
        smap = saliency_detector.get_saliency_map(img_rgb)

        # =====================================================
        # 1. TU MÉTODO (Propuesto: CNN + Interpolación)
        # =====================================================
        for th in qt_thresholds:
            start_t = time.time()
            
            compressor = QuadtreeCompressor(min_block_size=4, max_depth=10)
            # AQUÍ ESTABA EL ERROR: Pasamos img_rgb y smap explícitamente
            compressor.compress(img_rgb, smap, threshold=th, alpha=ALPHA_OPT)
            
            enc_time = time.time() - start_t
            
            # Peso Real: Usamos 'optimized' pora baja bpp con buena calidad
            compressed_bytes = codec.compress(compressor.root, (h, w), mode='optimized')
            bpp = (len(compressed_bytes) * 8) / pixels
            
            # Reconstrucción: Suave (Interpolada)
            rec = compressor.reconstruct((h, w))
            
            scores = metrics_engine.calculate_all(img_rgb, rec, saliency_map=smap)
            results.append({
                "Image": img_path.name, 
                "Method": "Ours (optimized)", 
                "Param": th, 
                "BPP": bpp, 
                "Time_s": enc_time,
                **scores
            })
            print(f"  [Ours Th={th}] BPP: {bpp:.3f} | LPIPS: {scores['lpips']:.4f}")

        # =====================================================
        # 2. STANDARD QUADTREE (Baseline: Sin IA + Bloques)
        # =====================================================
        for th in qt_thresholds:
            start_t = time.time()
            
            compressor = QuadtreeCompressor(min_block_size=4, max_depth=10)
            # Alpha = 0.0 apaga la influencia de la CNN (Saliencia no afecta el umbral)
            compressor.compress(img_rgb, smap, threshold=th, alpha=0.0)
            
            enc_time = time.time() - start_t
            
            # Peso Real: Usamos 'flat' porque un QT estándar guarda 1 color/hoja
            compressed_bytes = codec.compress(compressor.root, (h, w), mode='flat')
            bpp = (len(compressed_bytes) * 8) / pixels
            
            # Reconstrucción: Bloques (Minecraft style)
            rec = compressor.reconstruct_blocks((h, w))
            
            scores = metrics_engine.calculate_all(img_rgb, rec, saliency_map=smap)
            results.append({
                "Image": img_path.name, 
                "Method": "Standard QT", 
                "Param": th, 
                "BPP": bpp, 
                "Time_s": enc_time,
                **scores
            })
        
        # =====================================================
        # 2.5. STANDARD QT + INTERPOLACIÓN (El "Control" Científico)
        # =====================================================
        # Hipótesis: ¿La interpolación salva al Standard QT? 
        # Si tu método (IA) le gana a este, pruebas que la Saliencia es vital.
        for th in qt_thresholds:
            # Usamos alpha=0.0 (Varianza, sin IA)
            compressor = QuadtreeCompressor(min_block_size=4, max_depth=10)
            compressor.compress(img_rgb, smap, threshold=th, alpha=0.0)
            
            # IMPORTANTE: Usamos mode='gradient' porque para interpolar necesitamos 4 colores.
            # Esto pesará más (12 bytes/hoja), igual que tu método propuesto.
            compressed_bytes = codec.compress(compressor.root, (h, w), mode='optimized')
            bpp = (len(compressed_bytes) * 8) / pixels
            
            # Reconstrucción: SUAVE (Interpolada)
            rec = compressor.reconstruct((h, w))
            
            scores = metrics_engine.calculate_all(img_rgb, rec, saliency_map=smap)
            results.append({
                "Image": img_path.name, 
                "Method": "Standard QT (Interp)", # Nombre clave
                "Param": th, 
                "BPP": bpp, 
                "Time_s": 0,
                **scores
            })

        # =====================================================
        # 3. ABLACIÓN (Tu Método pero visualizado como Bloques)
        # =====================================================
        # Esto sirve para demostrar cuánto ganamos solo por interpolar
        for th in qt_thresholds:
            compressor = QuadtreeCompressor(min_block_size=4, max_depth=10)
            compressor.compress(img_rgb, smap, threshold=th, alpha=ALPHA_OPT) # Con IA
            
            # Si reconstruimos bloques, guardamos como 'flat' para ser justos en BPP vs Standard
            compressed_bytes = codec.compress(compressor.root, (h, w), mode='flat')
            bpp = (len(compressed_bytes) * 8) / pixels
            
            rec = compressor.reconstruct_blocks((h, w)) # Visualización Bloques
            
            scores = metrics_engine.calculate_all(img_rgb, rec, saliency_map=smap)
            results.append({
                "Image": img_path.name, 
                "Method": "Ours (Ablation-Blocks)", 
                "Param": th, 
                "BPP": bpp, 
                "Time_s": 0, # Ya medido arriba
                **scores
            })

        # =====================================================
        # 4 & 5. COMPETENCIA INDUSTRIAL (JPEG / WebP)
        # =====================================================
        for q in jpeg_qualities:
            temp = "temp_eval.jpg"
            start_t = time.time()
            cv2.imwrite(temp, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            enc_time = time.time() - start_t
            
            bpp = (os.path.getsize(temp) * 8) / pixels
            rec = cv2.cvtColor(cv2.imread(temp), cv2.COLOR_BGR2RGB)
            
            scores = metrics_engine.calculate_all(img_rgb, rec, saliency_map=smap)
            results.append({
                "Image": img_path.name, "Method": "JPEG", "Param": q, 
                "BPP": bpp, "Time_s": enc_time, **scores
            })

        for q in webp_qualities:
            temp = "temp_eval.webp"
            start_t = time.time()
            cv2.imwrite(temp, img_bgr, [cv2.IMWRITE_WEBP_QUALITY, q])
            enc_time = time.time() - start_t
            
            bpp = (os.path.getsize(temp) * 8) / pixels
            rec = cv2.cvtColor(cv2.imread(temp), cv2.COLOR_BGR2RGB)
            
            scores = metrics_engine.calculate_all(img_rgb, rec)
            results.append({
                "Image": img_path.name, "Method": "WebP", "Param": q, 
                "BPP": bpp, "Time_s": enc_time, **scores
            })

    # Limpieza
    if os.path.exists("temp_eval.jpg"): os.remove("temp_eval.jpg")
    if os.path.exists("temp_eval.webp"): os.remove("temp_eval.webp")

    # Guardar Resultados
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Benchmark finalizado. Resultados en {output_csv}")

if __name__ == "__main__":
    evaluate()