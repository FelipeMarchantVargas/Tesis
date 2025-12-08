import optuna
import cv2
import numpy as np
import pickle
import zlib
from pathlib import Path
from src.quadtree import QuadtreeCompressor
from src.saliency import SaliencyDetector
from src.metrics import QualityMetrics

# --- CONFIGURACIÓN ---
# Usamos pocas imágenes para que la optimización sea rápida (1-2 minutos)
# Elige 3 imágenes representativas de tu carpeta data/kodak
TRAIN_IMAGES = ["data/kodak/kodim04.png", "data/kodak/kodim15.png", "data/kodak/kodim23.png"] 
MODEL_PATH = "models/u2net.pth"

# Inicializamos motores fuera del bucle para no recargar modelo cada vez
print("Cargando motores...")
saliency_detector = SaliencyDetector(weights_path=MODEL_PATH)
metrics_engine = QualityMetrics()

# Cacheamos las imágenes y mapas de saliencia en RAM para velocidad extrema
cache_data = []
for img_path_str in TRAIN_IMAGES:
    path = Path(img_path_str)
    if not path.exists():
        continue
    img_bgr = cv2.imread(str(path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Pre-calculamos el mapa de saliencia (es fijo, no depende de alpha)
    smap = saliency_detector.get_saliency_map(img_rgb)
    
    cache_data.append({
        'rgb': img_rgb,
        'smap': smap,
        'pixels': img_rgb.shape[0] * img_rgb.shape[1]
    })

print(f"Optimización lista con {len(cache_data)} imágenes en caché.")

def objective(trial):
    """
    Esta función es ejecutada por Optuna cientos de veces.
    Optuna sugiere valores para 'threshold' y 'alpha'.
    Nosotros devolvemos un 'score' (menor es mejor).
    """
    # 1. Sugerir Hiperparámetros
    # Rango de búsqueda para el threshold (ej. entre 10 y 200)
    threshold = trial.suggest_float("threshold", 10.0, 200.0)
    
    # Rango de búsqueda para alpha (ej. entre 0.0 y 1.0)
    # alpha 0.0 = Quadtree normal. alpha 1.0 = Máxima influencia de IA.
    alpha = trial.suggest_float("alpha", 0.0, 1.0)

    total_loss = 0.0

    # 2. Probar en el set de entrenamiento
    for data in cache_data:
        img_rgb = data['rgb']
        smap = data['smap']
        pixels = data['pixels']
        h, w = img_rgb.shape[:2]

        # A. Comprimir
        compressor = QuadtreeCompressor(min_block_size=4, max_depth=10)
        compressor.compress(img_rgb, smap, threshold=threshold, alpha=alpha)
        
        # B. Reconstruir
        rec_rgb = compressor.reconstruct((h, w))
        
        # C. Calcular BPP Real
        payload = {'leaves': compressor.leaves, 'shape': (h, w)}
        compressed = zlib.compress(pickle.dumps(payload), level=9)
        bpp = (len(compressed) * 8) / pixels

        # D. Calcular LPIPS (Calidad)
        scores = metrics_engine.calculate_all(img_rgb, rec_rgb)
        lpips_score = scores['lpips']

        # --- LA FÓRMULA MÁGICA (Función de Costo) ---
        # Queremos bajo LPIPS y bajo BPP.
        # Lambda (0.15) es cuánto nos importa el tamaño vs la calidad.
        # Si subes 0.15 a 0.5, Optuna preferirá archivos más pequeños.
        # Si bajas a 0.05, Optuna preferirá más calidad.
        loss = lpips_score + (0.15 * bpp)
        
        total_loss += loss

    # Devolvemos el promedio de pérdida
    return total_loss / len(cache_data)

if __name__ == "__main__":
    # Creamos el estudio
    study = optuna.create_study(direction="minimize")
    
    print("--- Iniciando búsqueda de hiperparámetros (50 trials) ---")
    # n_trials=50 es un buen número para empezar (tardará unos 5-10 mins)
    study.optimize(objective, n_trials=50)

    print("\n--- ¡Optimización Terminada! ---")
    print("Mejores parámetros encontrados:")
    print(study.best_params)
    print(f"Mejor Score: {study.best_value:.4f}")
    
    # Guardar resultados para análisis
    df = study.trials_dataframe()
    df.to_csv("results/optuna_results.csv")
    
    # Generar gráficos de importancia (¡Oro para la tesis!)
    try:
        fig1 = optuna.visualization.plot_param_importances(study)
        fig1.write_image("results/optuna_importance.png")
        
        fig2 = optuna.visualization.plot_contour(study, params=["alpha", "threshold"])
        fig2.write_image("results/optuna_contour.png")
        print("Gráficos guardados en results/")
    except Exception as e:
        print(f"No se pudieron guardar gráficos interactivos (falta kaleido?): {e}")