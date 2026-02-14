# Evaluación del Dataset con Parámetros Optimizados
# Este script recorre el dataset de Kodak, aplica tu método propuesto con los parámetros optimizados, y también evalúa el baseline y las referencias JPEG/WebP. Los resultados se guardan en un CSV para su posterior análisis y visualización.
import cv2
import pandas as pd
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Importaciones
from src.quadtree import QuadtreeCompressor
from src.codec import QuadtreeCodec
from src.saliency import SaliencyDetector
from src.metrics import QualityMetrics

def evaluate():
    # --- CONFIGURACIÓN OPTIMIZADA ---
    # Valores obtenidos de tu optimización
    FIXED_LOW_TH = 9.9987
    ALPHA_OPT    = 5.1542
    BETA_OPT     = 4.9655
    
    # Rutas
    dataset_path = Path("data/kodak")
    output_csv = "results/benchmark_results_final.csv" 
    model_path = "models/u2net.pth"
    
    # Puntos de operación (Lambda)
    # Cubrimos un rango amplio para generar curvas completas
    rdo_lambdas = [5, 10, 20, 40, 70, 110, 150, 200, 300, 500, 1000]
    
    # Referencias
    jpeg_qualities = [5, 10, 20, 30, 50, 70, 90]
    webp_qualities = [5, 10, 20, 30, 50, 70, 90]

    # Iniciar Motores
    print("Cargando modelos...")
    saliency_detector = SaliencyDetector(weights_path=model_path)
    metrics_engine = QualityMetrics()
    codec = QuadtreeCodec()
    
    if not dataset_path.exists():
        print(f"Error: No se encuentra {dataset_path}")
        return
        
    images = sorted(list(dataset_path.glob("*.png")))
    results = []

    print(f"Iniciando Benchmark Final...")
    print(f"Params: Th={FIXED_LOW_TH:.2f}, Alpha={ALPHA_OPT:.2f}, Beta={BETA_OPT:.2f}")

    for img_path in tqdm(images, desc="Procesando Dataset"):
        # Cargar Imagen
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        pixels = h_orig * w_orig
        
        # Mapa de Saliencia
        smap = saliency_detector.get_saliency_map(img_rgb)
        
        # --- 1. PROPOSED METHOD (Saliency RDO) ---
        for lam in rdo_lambdas:
            start_time = time.time()
            
            compressor = QuadtreeCompressor(min_block_size=8, max_depth=10)
            compressor.compress(
                img_rgb, smap, 
                threshold=FIXED_LOW_TH, 
                alpha=ALPHA_OPT, 
                lam=lam, 
                beta=BETA_OPT
            )
            
            # Recuperamos la calidad dinámica calculada por el compresor
            quality_used = compressor.dynamic_quality

            padded_shape = (compressor.root.h, compressor.root.w)
            
            # Pasamos esa calidad al codec
            bitstream = codec.compress(compressor.root, padded_shape, quality_factor=quality_used)
            
            bpp = (len(bitstream) * 8) / pixels
            t_enc = time.time() - start_time
            
            # Decodificación (Recibimos 3 valores ahora)
            root_dec, _, q_read = codec.decompress(bitstream)
            
            compressor.orig_h = h_orig
            compressor.orig_w = w_orig
            
            # Reconstruimos usando la calidad leída del archivo
            rec_img = compressor.reconstruct(root_dec, override_quality=q_read)
            
            m = metrics_engine.calculate_all(img_rgb, rec_img, saliency_map=smap)
            
            results.append({
                "Image": img_path.name,
                "Method": "Proposed (Saliency RDO)", 
                "Param": lam,
                "BPP": bpp,
                "Time": t_enc,
                "PSNR": m['psnr'],
                "SSIM": m['ssim'],
                "SW-SSIM": m['sw_ssim'],
                "LPIPS": m['lpips']
            })

        # --- 2. BASELINE (Standard RDO) ---
        for lam in rdo_lambdas:
            # Baseline usa Alpha=0, Beta=0 (Sin Saliencia)
            compressor = QuadtreeCompressor(min_block_size=8, max_depth=10)
            compressor.compress(
                img_rgb, smap, 
                threshold=FIXED_LOW_TH, 
                alpha=0.0, # Apagado
                lam=lam, 
                beta=0.0   # Apagado
            )
            
            # --- CORRECCIÓN AQUÍ ---
            # El Baseline TAMBIÉN debe usar calidad dinámica (depende de Lambda, no de saliencia)
            quality_used = compressor.dynamic_quality
            
            padded_shape = (compressor.root.h, compressor.root.w)
            
            # Pasamos calidad al codec
            bitstream = codec.compress(compressor.root, padded_shape, quality_factor=quality_used)
            
            bpp = (len(bitstream) * 8) / pixels
            
            # Desempaquetamos correctamente los 3 valores
            root_dec, _, q_read = codec.decompress(bitstream)
            
            compressor.orig_h = h_orig
            compressor.orig_w = w_orig
            
            # Reconstruimos con la calidad correcta
            rec_img = compressor.reconstruct(root_dec, override_quality=q_read)
            # -----------------------
            
            m = metrics_engine.calculate_all(img_rgb, rec_img, saliency_map=smap)
            
            results.append({
                "Image": img_path.name,
                "Method": "Baseline (Standard RDO)", 
                "Param": lam,
                "BPP": bpp,
                "Time": 0,
                "PSNR": m['psnr'],
                "SSIM": m['ssim'],
                "SW-SSIM": m['sw_ssim'],
                "LPIPS": m['lpips']
            })

        # --- 3. REFERENCIAS ---
        for q in jpeg_qualities:
            _, enc = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            bpp = len(enc) * 8 / pixels
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            m = metrics_engine.calculate_all(img_rgb, dec, saliency_map=smap)
            results.append({
                "Image": img_path.name, "Method": "JPEG", "Param": q, 
                "BPP": bpp, "Time": 0, "PSNR": m['psnr'], "SSIM": m['ssim'], "SW-SSIM": m['sw_ssim'], "LPIPS": m['lpips']
            })

        for q in webp_qualities:
            _, enc = cv2.imencode('.webp', img_bgr, [cv2.IMWRITE_WEBP_QUALITY, q])
            bpp = len(enc) * 8 / pixels
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            m = metrics_engine.calculate_all(img_rgb, dec, saliency_map=smap)
            results.append({
                "Image": img_path.name, "Method": "WebP", "Param": q, 
                "BPP": bpp, "Time": 0, "PSNR": m['psnr'], "SSIM": m['ssim'], "SW-SSIM": m['sw_ssim'], "LPIPS": m['lpips']
            })

    # Guardar
    df = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResultados guardados en {output_csv}")

if __name__ == "__main__":
    evaluate()