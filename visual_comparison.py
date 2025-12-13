import matplotlib
matplotlib.use('Agg') # Backend sin ventana
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Importaciones de tu proyecto
from src.quadtree import QuadtreeCompressor
from src.saliency import SaliencyDetector
from src.codec import QuadtreeCodec

def find_matching_jpeg(img_rgb, target_bpp, pixels):
    """Encuentra la calidad JPEG que iguala el BPP objetivo."""
    low, high = 1, 100
    best_q = 50
    min_diff = float('inf')
    best_rec = img_rgb.copy()
    final_bpp = 0
    
    # Búsqueda binaria rápida
    for _ in range(15):
        mid = (low + high) // 2
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, mid])
        curr_bpp = (len(buf) * 8) / pixels
        
        diff = abs(curr_bpp - target_bpp)
        if diff < min_diff:
            min_diff = diff
            best_q = mid
            final_bpp = curr_bpp
            best_rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            best_rec = cv2.cvtColor(best_rec, cv2.COLOR_BGR2RGB)
            
        if curr_bpp < target_bpp:
            low = mid + 1
        else:
            high = mid - 1
            
    return best_rec, best_q, final_bpp

def find_matching_std_qt(img_rgb, smap, target_bpp, pixels, codec):
    """
    Busca el threshold del Standard QT (Bloques) que iguale el BPP.
    Queremos comparar contra 'Standard QT' (Bloques) porque es el que se ve 'feo'.
    """
    low, high = 1.0, 300.0
    best_th = 50
    min_diff = float('inf')
    best_rec = img_rgb.copy()
    final_bpp = 0
    
    h, w = img_rgb.shape[:2]
    
    for _ in range(15):
        mid = (low + high) / 2
        comp = QuadtreeCompressor(min_block_size=4, max_depth=10)
        # Alpha 0.0 = Varianza Pura
        comp.compress(img_rgb, smap, threshold=mid, alpha=0.0) 
        
        # Standard QT usa 'flat' (1 color, 3 bytes)
        data = codec.compress(comp.root, (h, w), mode='flat')
        curr_bpp = (len(data) * 8) / pixels
        
        diff = abs(curr_bpp - target_bpp)
        if diff < min_diff:
            min_diff = diff
            best_th = mid
            final_bpp = curr_bpp
            best_rec = comp.reconstruct_blocks((h, w)) # Reconstrucción Bloques
        
        if curr_bpp > target_bpp:
            low = mid # Pesa mucho -> subir threshold (menos nodos)
        else:
            high = mid
            
    return best_rec, final_bpp

def generate_comparison():
    # --- CONFIGURACIÓN ---
    # Imagen sugerida: kodim05 (motos) o kodim17 (rostro mono)
    img_name = "kodim04.png" 
    img_path = Path(f"data/kodak/{img_name}")
    model_path = "models/u2net.pth"
    
    # Parámetros de TU método
    # Usamos un Threshold que nos de un BPP interesante (ej. ~0.5 - 0.8 BPP)
    ALPHA_OPT = 4.7557
    TARGET_TH = 25 # Ajusta esto si quieres más o menos calidad en la demo
    
    # Zoom Region (Ajustar según la imagen para ver detalle)
    # Para kodim05 (Motos):
    ZOOM_ROI = (100, 280, 150, 150) # Y, X, H, W
    
    print(f"Generando comparativa visual FINAL para {img_name}...")
    
    saliency = SaliencyDetector(model_path)
    codec = QuadtreeCodec()
    
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("Error: Imagen no encontrada.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    pixels = h * w
    smap = saliency.get_saliency_map(img_rgb)
    
    # 1. EJECUTAR TU MÉTODO (OPTIMIZADO)
    print("1. Comprimiendo con Proposed (Saliency + Optimized Codec)...")
    qt_ours = QuadtreeCompressor()
    qt_ours.compress(img_rgb, smap, threshold=TARGET_TH, alpha=ALPHA_OPT)
    
    # IMPORTANTE: Usamos 'optimized' (YCbCr 4:2:0)
    bytes_ours = codec.compress(qt_ours.root, (h, w), mode='optimized')
    bpp_ref = (len(bytes_ours) * 8) / pixels
    rec_ours = qt_ours.reconstruct((h, w))
    
    print(f"   -> BPP Objetivo establecido: {bpp_ref:.3f}")

    # 2. STANDARD QT (Matching BPP)
    print("2. Buscando Standard QT (Blocks) match...")
    rec_std, bpp_std = find_matching_std_qt(img_rgb, smap, bpp_ref, pixels, codec)
    
    # 3. JPEG (Matching BPP)
    print("3. Buscando JPEG match...")
    rec_jpg, q_jpg, bpp_jpg = find_matching_jpeg(img_rgb, bpp_ref, pixels)

    # --- GRAFICAR ---
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    
    # Lista de imágenes
    imgs = [
        ('Original', img_rgb),
        (f'Proposed (Saliency)\nBPP: {bpp_ref:.2f}', rec_ours),
        (f'Baseline (Std QT)\nBPP: {bpp_std:.2f}', rec_std),
        (f'JPEG (Q={q_jpg})\nBPP: {bpp_jpg:.2f}', rec_jpg)
    ]
    
    y, x, zh, zw = ZOOM_ROI
    
    # Filas
    for i, (title, img) in enumerate(imgs):
        # Fila 1: Full
        ax_full = axes[0, i]
        ax_full.imshow(img)
        ax_full.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax_full.axis('off')
        # Rectángulo
        rect = plt.Rectangle((x, y), zw, zh, linewidth=3, edgecolor='#FF3333', facecolor='none')
        ax_full.add_patch(rect)
        
        # Fila 2: Zoom
        ax_zoom = axes[1, i]
        crop = img[y:y+zh, x:x+zw]
        # Interpolación 'nearest' para ver los pixeles reales (bloques)
        ax_zoom.imshow(crop, interpolation='nearest') 
        ax_zoom.axis('off')
        
        # Borde al zoom
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor('#FF3333')
            spine.set_linewidth(2)

    plt.tight_layout()
    os.makedirs("results/visuals", exist_ok=True)
    save_path = f"results/visuals/final_comparison_optimized04.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparativa Final guardada en: {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_comparison()