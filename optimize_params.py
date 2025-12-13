import optuna
import cv2
import numpy as np
import pandas as pd
import os
import tqdm
from pathlib import Path

# Importaciones de tu proyecto
# Asegúrate de que src.quadtree tenga la lógica de 'np.exp' en _recursive_split
from src.quadtree import QuadtreeCompressor
from src.saliency import SaliencyDetector
from src.metrics import QualityMetrics
from src.codec import QuadtreeCodec

def run_optimization():
    # --- CONFIGURACIÓN ---
    DATASET_DIR = Path("data/kodak")
    MODEL_PATH = "models/u2net.pth"
    
    # N_TRIALS: 50 es suficiente para encontrar una buena zona.
    N_TRIALS = 50 
    
    # Lambda (λ): El balance entre Calidad y Peso.
    # J = LPIPS + λ * BPP
    # Valor empírico: 0.07 prioriza una compresión media-alta con buena calidad.
    LAMBDA = 0.07 

    print("--- Configurando Optimización Rate-Distortion (Lagrangiana) ---")

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

    # 2. Cachear Dataset (Para no re-leer disco ni re-calcular Saliencia)
    # Seleccionamos un subconjunto representativo si son muchas imágenes (ej. 10)
    # para que la optimización sea rápida. Si tienes GPU, usa todas.
    subset_paths = image_paths[:10] # Usamos las primeras 10 para calibrar rápido
    cache_data = []
    
    print(f"Pre-cargando {len(subset_paths)} imágenes en RAM...")
    for path in tqdm.tqdm(subset_paths):
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Mapa de Saliencia (Constante durante toda la optimización)
        smap = saliency_detector.get_saliency_map(img_rgb)
        
        cache_data.append({
            'rgb': img_rgb,
            'smap': smap,
            'shape': (h, w),
            'pixels': h * w
        })

    # --- FUNCIÓN OBJETIVO ---
    def objective(trial):
        # 1. Sugerencia de Hiperparámetros
        
        # Threshold: Rango amplio. De 10 (muy detallado) a 120 (muy comprimido)
        threshold = trial.suggest_float("threshold", 10.0, 120.0)
        
        # Alpha (Exponencial): Rango ajustado a la fórmula np.exp(-alpha * S)
        # 0.0 = Sin efecto. 5.0 = Efecto muy agresivo en zonas salientes.
        alpha = trial.suggest_float("alpha", 0.0, 5.0)

        total_cost = 0.0

        # 2. Evaluar sobre el dataset cacheado
        for data in cache_data:
            img_rgb = data['rgb']
            smap = data['smap']
            h, w = data['shape']
            
            # A. Compresión
            compressor = QuadtreeCompressor(min_block_size=4, max_depth=10)
            # Nota: Asegúrate que compressor.compress use internamente la fórmula exponencial
            compressor.compress(img_rgb, smap, threshold=threshold, alpha=alpha)
            
            # B. Cálculo de BPP REAL (Usando mode='gradient' porque es tu método propuesto)
            try:
                # mode='gradient' guarda 12 bytes/hoja (interpolación)
                compressed_bytes = codec.compress(compressor.root, (h, w), mode='gradient')
                bpp = (len(compressed_bytes) * 8) / data['pixels']
            except Exception as e:
                # Si falla algo (ej. árbol vacío), penalizamos infinito
                return float('inf')

            # C. Reconstrucción y Calidad
            rec_rgb = compressor.reconstruct((h, w))
            
            # Usamos LPIPS como métrica principal de distorsión (D)
            # LPIPS bajo es mejor (0.0 es idéntico)
            scores = metrics_engine.calculate_all(img_rgb, rec_rgb)
            lpips_val = scores['lpips']
            
            # --- COSTO LAGRANGIANO ---
            # Optuna minimizará este valor.
            # Buscamos el mejor compromiso entre calidad (LPIPS) y peso (BPP)
            loss = lpips_val + (LAMBDA * bpp)
            
            total_cost += loss

        # Promedio del costo en el dataset
        return total_cost / len(cache_data)

    # --- EJECUCIÓN ---
    # Usamos TPE (Tree-structured Parzen Estimator) que es el default de Optuna, es excelente.
    study = optuna.create_study(direction="minimize")
    
    print("\nIniciando búsqueda de hiperparámetros...")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*40)
    print(" RESULTADOS DE OPTIMIZACIÓN")
    print("="*40)
    print(f"Mejor Loss (J): {study.best_value:.4f}")
    print("Mejores Parámetros:")
    print(f"  Threshold (Base): {study.best_params['threshold']:.4f}")
    print(f"  Alpha (Fuerza):   {study.best_params['alpha']:.4f}")
    
    # Guardar
    os.makedirs("results", exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv("results/optuna_results_robust.csv", index=False)
    print("Log guardado en results/optuna_results_robust.csv")

if __name__ == "__main__":
    run_optimization()