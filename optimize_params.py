import optuna
import cv2
import numpy as np
import pandas as pd
import os
import tqdm
from pathlib import Path

# Importaciones de tu proyecto
from src.quadtree import QuadtreeCompressor
from src.saliency import SaliencyDetector
from src.metrics import QualityMetrics
from src.codec import QuadtreeCodec

def run_optimization():
    # --- CONFIGURACIÓN ---
    DATASET_DIR = Path("data/kodak")
    MODEL_PATH = "models/u2net.pth"
    
    # N_TRIALS: 50 iteraciones suelen ser suficientes para encontrar un buen óptimo local.
    N_TRIALS = 50
    
    # LAMBDA FIJO para la optimización.
    # Usamos un valor intermedio (ej. 40) que representa un balance típico.
    # Optimizaremos los parámetros estructurales para que funcionen bien en este punto de operación.
    FIXED_LAMBDA_RDO = 40.0 
    
    # Peso del BPP en la función de costo de Optuna
    # Objetivo = (1.0 - SW_SSIM) + (COSTO_BPP * BPP)
    # Buscamos maximizar SW-SSIM manteniendo el BPP bajo control.
    COSTO_BPP_OPTUNA = 0.1 

    print("--- Optimizando Hiperparámetros (Threshold, Alpha, Beta) ---")

    if not DATASET_DIR.exists():
        print(f"ERROR: No existe {DATASET_DIR}")
        return

    image_paths = sorted(list(DATASET_DIR.glob("*.png")))
    if not image_paths:
        print("ERROR: Carpeta vacía.")
        return

    # 1. Inicializar Motores
    print("Cargando modelos...")
    saliency_detector = SaliencyDetector(weights_path=MODEL_PATH)
    metrics_engine = QualityMetrics()
    codec = QuadtreeCodec()

    # 2. Cachear Dataset (Primeras 8 imágenes para velocidad)
    # Usar un subconjunto acelera la búsqueda y generaliza bien para parámetros estructurales.
    subset_paths = image_paths[:8] 
    cache_data = []
    
    print(f"Pre-cargando {len(subset_paths)} imágenes para optimización rápida...")
    for path in tqdm.tqdm(subset_paths):
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Mapa de Saliencia
        smap = saliency_detector.get_saliency_map(img_rgb)
        
        cache_data.append({
            'rgb': img_rgb,
            'smap': smap,
            'shape': (h, w),
            'pixels': h * w
        })

    # --- FUNCIÓN OBJETIVO ---
    def objective(trial):
        # 1. Sugerencia de Hiperparámetros (Espacio de Búsqueda)
        
        # Threshold: Permitimos bajar a 0.0 para que la DCT (que requiere 8x8) tenga oportunidad de activarse.
        # Un threshold bajo fuerza la división inicial, dejando la decisión final al RDO.
        threshold = trial.suggest_float("threshold", 0.0, 10.0)
        
        # Alpha (Top-Down): Sensibilidad inicial a la saliencia en la etapa de split.
        alpha = trial.suggest_float("alpha", 0.0, 8.0)
        
        # Beta (Bottom-Up): Protección de saliencia en RDO.
        # Beta=0 es RDO estándar. Beta alto protege mucho las zonas de interés.
        beta = trial.suggest_float("beta", 0.0, 10.0)

        total_loss = 0.0

        for data in cache_data:
            img_rgb = data['rgb']
            smap = data['smap']
            
            # A. Compresión con los parámetros sugeridos
            # CRÍTICO: min_block_size=8 para habilitar DCT.
            # Si bajamos a 4, la DCT se desactiva y optimizamos parámetros erróneos.
            compressor = QuadtreeCompressor(min_block_size=8, max_depth=10)
            
            try:
                compressor.compress(
                    img_rgb, smap, 
                    threshold=threshold, 
                    alpha=alpha, 
                    lam=FIXED_LAMBDA_RDO, 
                    beta=beta             
                )
                
                # B. Calcular BPP Real (Multi-Mode)
                # CRÍTICO: Usar dimensiones del árbol PADDEADO (root.h, root.w).
                # Si pasamos las dimensiones originales, el codec fallará al deserializar.
                padded_shape = (compressor.root.h, compressor.root.w)
                compressed_bytes = codec.compress(compressor.root, padded_shape)
                
                # BPP se calcula sobre los píxeles originales útiles (data['pixels'])
                bpp = (len(compressed_bytes) * 8) / data['pixels']
                
                # C. Reconstrucción Directa
                # El método reconstruct() ya maneja internamente el recorte del padding.
                rec_img = compressor.reconstruct(compressor.root)
                
                # D. Calcular Métricas
                # Pasamos smap para calcular SW-SSIM correctamente.
                scores = metrics_engine.calculate_all(img_rgb, rec_img, saliency_map=smap)
                
                # E. Función de Costo para Optuna
                # Maximizamos SW-SSIM (1 - SW-SSIM) minimizando BPP
                metric_loss = 1.0 - scores['sw_ssim']
                rate_loss = COSTO_BPP_OPTUNA * bpp
                
                total_loss += (metric_loss + rate_loss)
                
            except Exception as e:
                # Si falla una configuración (ej. BPP explota o error numérico), penalizamos fuerte.
                return float('inf')

        # Retornamos el promedio del costo en el dataset
        return total_loss / len(cache_data)

    # --- EJECUCIÓN ---
    study = optuna.create_study(direction="minimize")
    
    print("\nIniciando búsqueda de la 'Trinidad' de parámetros...")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*50)
    print(" RESULTADOS OPTIMIZADOS PARA TESIS")
    print("="*50)
    print(f"Mejor Loss: {study.best_value:.4f}")
    print("--- COPIA ESTOS VALORES EN evaluate_dataset.py ---")
    print(f"FIXED_LOW_TH = {study.best_params['threshold']:.4f}")
    print(f"ALPHA_OPT    = {study.best_params['alpha']:.4f}")
    print(f"BETA_OPT     = {study.best_params['beta']:.4f}")
    print("="*50)
    
    # Guardar CSV detallado para análisis posterior
    os.makedirs("results", exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv("results/optuna_optimization_log.csv", index=False)
    print("Log guardado en results/optuna_optimization_log.csv")

if __name__ == "__main__":
    run_optimization()