import cv2
import numpy as np
from src.quadtree import QuadtreeCompressor
from src.saliency import SaliencyDetector

def generate_comparison():
    # Configuración de la "Foto Ganadora"
    img_path = "data/kodak/kodim04.png" # Usamos la chica del sombrero (clásica) o la que prefieras
    model_path = "models/u2net.pth"
    
    # Parámetros Óptimos (Los que nos dio Optuna o cercanos)
    best_th = 5.0
    best_alpha = 0.17
    
    # JPEG para comparar (Calidad muy baja para forzar artefactos)
    jpeg_quality = 30

    print(f"Generando comparativa con {img_path}...")
    
    # 1. Cargar
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 2. Tu Método
    saliency = SaliencyDetector(weights_path=model_path).get_saliency_map(img_rgb)
    compressor = QuadtreeCompressor(min_block_size=4, max_depth=10)
    compressor.compress(img_rgb, saliency, threshold=best_th, alpha=best_alpha)
    rec_ours = compressor.reconstruct((h, w))
    rec_ours_bgr = cv2.cvtColor(rec_ours, cv2.COLOR_RGB2BGR)

    # 3. JPEG
    cv2.imwrite("temp_comp.jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    rec_jpeg_bgr = cv2.imread("temp_comp.jpg")

    # 4. Montaje (Tríptico)
    # Recortamos un pedazo interesante (Crop) para que se vean los detalles
    # Centro de la imagen 256x256
    cy, cx = h // 2, w // 2
    y1, y2 = cy - 128, cy + 128
    x1, x2 = cx - 128, cx + 128
    
    crop_orig = img_bgr[y1:y2, x1:x2]
    crop_jpeg = rec_jpeg_bgr[y1:y2, x1:x2]
    crop_ours = rec_ours_bgr[y1:y2, x1:x2]

    # Unir horizontalmente
    final_comp = np.hstack([crop_orig, crop_jpeg, crop_ours])
    
    # Guardar
    cv2.imwrite("results/visual_comparison_HIGH_QUALITY.png", final_comp)
    print("¡Listo! Imagen guardada en results/visual_comparison_HIGH_QUALITY.png")
    print("Orden: Original | JPEG (Bloques) | Tuyo (Suave)")

if __name__ == "__main__":
    generate_comparison()