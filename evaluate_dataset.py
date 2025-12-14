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
    
    # PARAMETROS CIENTÍFICOS
    ALPHA_OPT = 4.7557 
    
    # NUEVO: Barrido de Lambdas para RDO (Rate-Distortion Optimization)
    # Valores bajos (1.0) = Alta Calidad / Poco peso a los bits.
    # Valores altos (150.0) = Alta Compresión / Ahorrar bits es prioridad.
    rdo_lambdas = [1, 5, 10, 20, 40, 70, 110, 150]
    
    # Umbral inicial FIJO y BAJO.
    # Queremos sobre-segmentar al inicio para que el RDO tenga libertad de podar óptimamente.
    FIXED_LOW_TH = 5 
    
    # Calidades de referencia estándar
    jpeg_qualities = [10, 20, 40, 60, 80, 95]
    webp_qualities = [10, 20, 40, 60, 80, 95]

    # Inicializar Modelos
    saliency_detector = SaliencyDetector(weights_path=model_path)
    metrics = QualityMetrics()
    codec = QuadtreeCodec()
    
    # Obtener imágenes
    if not dataset_path.exists():
        print(f"Error: No se encuentra {dataset_path}")
        return
        
    images = sorted(list(dataset_path.glob("*.png")))
    results = []

    print(f"Iniciando evaluación en {len(images)} imágenes...")
    print(f"Modo RDO Activo. Variando Lambda: {rdo_lambdas}")

    for img_path in images:
        print(f"\nProcesando: {img_path.name}")
        
        # Cargar Imagen
        img_rgb = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Generar Mapa de Saliencia (Una vez por imagen)
        smap = saliency_detector.get_saliency_map(img_rgb)
        
        # --- 1. OUR METHOD (Saliency RDO) ---
        # Tu Tesis: Usa RDO con protección de Saliencia (beta=2.0)
        compressor = QuadtreeCompressor(min_block_size=4, max_depth=12)
        
        for lam in rdo_lambdas:
            start_time = time.time()
            
            # Compresión: Threshold bajo, Beta activado (2.0), Lambda variable
            compressor.compress(img_rgb, smap, threshold=FIXED_LOW_TH, alpha=ALPHA_OPT, lam=lam, beta=2.0)
            
            # Codificación
            compressed_bytes = codec.compress(compressor.root, (h, w), mode='optimized')
            bpp = len(compressed_bytes) * 8 / (h * w)
            t_enc = time.time() - start_time
            
            # Reconstrucción y Métricas
            # Reconstrucción y Métricas
            rec_root, _ = codec.decompress(compressed_bytes) # 1. Captura la forma (h, w)
            rec_img = compressor.reconstruct(rec_root)    # 2. Pásala como primer argumento
            
            # --- DEBUG: GUARDAR IMAGEN ---
            if lam == 1: # Solo guardar la primera para no llenar el disco
                debug_path = f"debug_reconstruct_{lam}.png"
                # Convertir a BGR para OpenCV
                cv2.imwrite(debug_path, cv2.cvtColor(rec_img, cv2.COLOR_RGB2BGR))
                print(f"Imagen de debug guardada en: {debug_path}")
            # -----------------------------

            m = metrics.calculate_all(img_rgb, rec_img)
            
            res = {
                "Image": img_path.name,
                "Method": "Ours (Saliency RDO)", # Nombre actualizado
                "Param": lam, # Ahora el parámetro es Lambda
                "BPP": bpp,
                "Time_s": t_enc,
                **m
            }
            results.append(res)
            print(f"  [Ours RDO] Lam={lam}: BPP={bpp:.3f}, SSIM={m['ssim']:.3f}")

        # --- 2. BASELINE (Standard RDO) ---
        # Control: Usa el mismo algoritmo pero SIN mirar la saliencia (beta=0.0)
        # Esto demuestra que tu mejora viene de la saliencia y no solo del RDO.
        for lam in rdo_lambdas:
            start_time = time.time()
            
            # Compresión: Threshold bajo, Beta apagado (0.0)
            compressor.compress(img_rgb, smap, threshold=FIXED_LOW_TH, alpha=0.0, lam=lam, beta=0.0)
            
            compressed_bytes = codec.compress(compressor.root, (h, w), mode='optimized')
            bpp = len(compressed_bytes) * 8 / (h * w)
            t_enc = time.time() - start_time
            
            rec_root, _ = codec.decompress(compressed_bytes)
            rec_img = compressor.reconstruct(rec_root)
            
            m = metrics.calculate_all(img_rgb, rec_img)
            
            res = {
                "Image": img_path.name,
                "Method": "Standard RDO (No Saliency)",
                "Param": lam,
                "BPP": bpp,
                "Time_s": t_enc,
                **m
            }
            results.append(res)

        # --- 3. JPEG (Referencia) ---
        for q in jpeg_qualities:
            start_time = time.time()
            _, enc = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
            bpp = len(enc) * 8 / (h * w)
            t_enc = time.time() - start_time
            
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            
            m = metrics.calculate_all(img_rgb, dec)
            results.append({
                "Image": img_path.name,
                "Method": "JPEG",
                "Param": q,
                "BPP": bpp,
                "Time_s": t_enc,
                **m
            })

        # --- 4. WebP (Referencia) ---
        for q in webp_qualities:
            start_time = time.time()
            _, enc = cv2.imencode('.webp', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, q])
            bpp = len(enc) * 8 / (h * w)
            t_enc = time.time() - start_time
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            m = metrics.calculate_all(img_rgb, dec)
            results.append({
                "Image": img_path.name,
                "Method": "WebP",
                "Param": q,
                "BPP": bpp,
                "Time_s": t_enc,
                **m
            })

    # Guardar CSV final
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nEvaluación completa. Resultados guardados en {output_csv}")

if __name__ == "__main__":
    evaluate()