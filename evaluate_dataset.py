# Evaluación del Dataset con Parámetros Optimizados y Profiling Completo
import cv2
import pandas as pd
import time
import numpy as np
import tracemalloc
import torch  # <--- FALTABA ESTE IMPORT
from pathlib import Path
from tqdm import tqdm

# Importaciones
from src.quadtree import QuadtreeCompressor
from src.codec import QuadtreeCodec
from src.saliency import SaliencyDetector
from src.metrics import QualityMetrics

# --- HELPER DE PROFILING ---
class Profiler:
    """ Mide tiempo y memoria pico de un bloque de código """
    def __init__(self):
        self.start_time = 0
        self.peak_memory = 0 # En MB
    
    def __enter__(self):
        # Limpiamos caché de GPU si está disponible para medir memoria real
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        tracemalloc.start()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory = peak / (1024 * 1024) # Convertir a MB
        self.duration = self.end_time - self.start_time

def evaluate():
    # --- CONFIGURACIÓN OPTIMIZADA ---
    FIXED_LOW_TH = 9.9987
    ALPHA_OPT    = 5.1542
    BETA_OPT     = 4.9655
    
    # Rutas
    dataset_path = Path("data/kodak")
    output_csv = "results/benchmark_results_full_metrics.csv" 
    model_path = "models/u2net.pth"
    
    # Puntos de operación (Lambda)
    # Usa tu lista completa aquí si quieres más resolución en las curvas
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

    print(f"Iniciando Benchmark Completo...")
    print(f"Params: Th={FIXED_LOW_TH:.2f}, Alpha={ALPHA_OPT:.2f}, Beta={BETA_OPT:.2f}")

    for img_path in tqdm(images, desc="Procesando Dataset"):
        # Cargar Imagen
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        pixels = h_orig * w_orig
        
        # Mapa de Saliencia (Se calcula una vez por imagen para la métrica SW-SSIM)
        smap_metrics = saliency_detector.get_saliency_map(img_rgb)
        
        # --- 1. PROPOSED METHOD (Saliency RDO) ---
        for lam in rdo_lambdas:
            compressor = QuadtreeCompressor(min_block_size=8, max_depth=10)
            
            # A) ENCODING PROFILE
            with Profiler() as p_enc:
                # Incluimos get_saliency_map en el tiempo si es parte del pipeline real
                # Si prefieres medir solo la compresión asumiendo que el mapa ya existe, sácalo del bloque.
                smap = saliency_detector.get_saliency_map(img_rgb) 
                
                compressor.compress(
                    img_rgb, smap, 
                    threshold=FIXED_LOW_TH, 
                    alpha=ALPHA_OPT, 
                    lam=lam, 
                    beta=BETA_OPT
                )
                quality_used = compressor.dynamic_quality
                padded_shape = (compressor.root.h, compressor.root.w)
                bitstream = codec.compress(compressor.root, padded_shape, quality_factor=quality_used)
            
            # Stats de compresión
            bytes_size = len(bitstream)
            bpp = (bytes_size * 8) / pixels
            cr = (pixels * 3) / bytes_size if bytes_size > 0 else 0
            
            # B) DECODING PROFILE
            with Profiler() as p_dec:
                root_dec, _, q_read = codec.decompress(bitstream)
                compressor.orig_h = h_orig
                compressor.orig_w = w_orig
                rec_img = compressor.reconstruct(root_dec, override_quality=q_read)

            # Metrics
            m = metrics_engine.calculate_all(img_rgb, rec_img, saliency_map=smap_metrics)
            
            results.append({
                "Image": img_path.name,
                "Method": "Proposed (Saliency RDO)", 
                "Param": lam,
                "BPP": bpp,
                "CR": cr, 
                "Enc_Time(s)": p_enc.duration,
                "Dec_Time(s)": p_dec.duration,
                "Enc_Mem(MB)": p_enc.peak_memory,
                "Dec_Mem(MB)": p_dec.peak_memory,
                "PSNR": m['psnr'],
                "SSIM": m['ssim'],
                "SW-SSIM": m['sw_ssim'],
                "MS-SSIM": m['ms_ssim'],
                "LPIPS": m['lpips'],
                "VIF": m['vif']
            })

        # --- 2. BASELINE (Standard RDO) ---
        for lam in rdo_lambdas:
            compressor = QuadtreeCompressor(min_block_size=8, max_depth=10)
            
            with Profiler() as p_enc:
                # Baseline NO usa saliencia para comprimir (Alpha=0, Beta=0)
                # Pasamos smap solo para cumplir la firma, pero los pesos lo ignoran
                compressor.compress(
                    img_rgb, smap_metrics, 
                    threshold=FIXED_LOW_TH, 
                    alpha=0.0, 
                    lam=lam, 
                    beta=0.0
                )
                quality_used = compressor.dynamic_quality
                padded_shape = (compressor.root.h, compressor.root.w)
                bitstream = codec.compress(compressor.root, padded_shape, quality_factor=quality_used)
            
            bytes_size = len(bitstream)
            bpp = (bytes_size * 8) / pixels
            cr = (pixels * 3) / bytes_size if bytes_size > 0 else 0

            with Profiler() as p_dec:
                root_dec, _, q_read = codec.decompress(bitstream)
                compressor.orig_h = h_orig
                compressor.orig_w = w_orig
                rec_img = compressor.reconstruct(root_dec, override_quality=q_read)
            
            m = metrics_engine.calculate_all(img_rgb, rec_img, saliency_map=smap_metrics)
            
            results.append({
                "Image": img_path.name,
                "Method": "Baseline (Standard RDO)", 
                "Param": lam,
                "BPP": bpp,
                "CR": cr,
                "Enc_Time(s)": p_enc.duration,
                "Dec_Time(s)": p_dec.duration,
                "Enc_Mem(MB)": p_enc.peak_memory,
                "Dec_Mem(MB)": p_dec.peak_memory,
                "PSNR": m['psnr'],
                "SSIM": m['ssim'],
                "SW-SSIM": m['sw_ssim'],
                "MS-SSIM": m['ms_ssim'],
                "LPIPS": m['lpips'],
                "VIF": m['vif']
            })

        # --- 3. JPEG ---
        for q in jpeg_qualities:
            with Profiler() as p_enc:
                _, enc = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            
            bytes_size = len(enc)
            bpp = bytes_size * 8 / pixels
            cr = (pixels * 3) / bytes_size if bytes_size > 0 else 0
            
            with Profiler() as p_dec:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            
            m = metrics_engine.calculate_all(img_rgb, dec, saliency_map=smap_metrics)
            results.append({
                "Image": img_path.name, "Method": "JPEG", "Param": q,
                "BPP": bpp, "CR": cr,
                "Enc_Time(s)": p_enc.duration, "Dec_Time(s)": p_dec.duration,
                "Enc_Mem(MB)": p_enc.peak_memory, "Dec_Mem(MB)": p_dec.peak_memory,
                "PSNR": m['psnr'], "SSIM": m['ssim'], "SW-SSIM": m['sw_ssim'], 
                "MS-SSIM": m['ms_ssim'], "LPIPS": m['lpips'], "VIF": m['vif']
            })

        # --- 4. WebP ---
        for q in webp_qualities:
            with Profiler() as p_enc:
                _, enc = cv2.imencode('.webp', img_bgr, [cv2.IMWRITE_WEBP_QUALITY, q])
            
            bytes_size = len(enc)
            bpp = bytes_size * 8 / pixels
            cr = (pixels * 3) / bytes_size if bytes_size > 0 else 0
            
            with Profiler() as p_dec:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            
            m = metrics_engine.calculate_all(img_rgb, dec, saliency_map=smap_metrics)
            results.append({
                "Image": img_path.name, "Method": "WebP", "Param": q,
                "BPP": bpp, "CR": cr,
                "Enc_Time(s)": p_enc.duration, "Dec_Time(s)": p_dec.duration,
                "Enc_Mem(MB)": p_enc.peak_memory, "Dec_Mem(MB)": p_dec.peak_memory,
                "PSNR": m['psnr'], "SSIM": m['ssim'], "SW-SSIM": m['sw_ssim'], 
                "MS-SSIM": m['ms_ssim'], "LPIPS": m['lpips'], "VIF": m['vif']
            })

    # Guardar
    df = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResultados guardados en {output_csv}")

if __name__ == "__main__":
    evaluate()