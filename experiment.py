import cv2
import numpy as np
import os
import time

# Asegúrate de que las importaciones coincidan con la estructura de tus carpetas (ej. from src.codec import ...)
from src.codec import QuadtreeCodec
from src.quadtree import QuadtreeCompressor
from src.saliency import SaliencyDetector
from src.metrics import QualityMetrics

def run_compression_experiment(image_path: str, output_dir: str):
    """
    Ejecuta el pipeline de compresión a diferentes tasas y guarda los resultados.
    """
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen en {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cargar imagen
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    print(f"Imagen original cargada: {w}x{h} px")
    
    # 2. Inicializar módulos
    # Si tienes los pesos de U2NET, pasa la ruta en weights_path='ruta/u2net.pth'
    saliency_detector = SaliencyDetector(device='auto') 
    compressor = QuadtreeCompressor(min_block_size=8, max_depth=8)
    codec = QuadtreeCodec()
    metrics = QualityMetrics()
    
    # 3. Obtener Mapa de Saliencia (Se calcula una sola vez)
    print("Calculando mapa de saliencia...")
    saliency_map = saliency_detector.get_saliency_map(img_rgb)
    
    # Guardar mapa de saliencia visualmente
    saliency_vis = (saliency_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "00_saliency_map.png"), saliency_vis)
    
    # 4. Definir Niveles de Compresión
    # Tupla: (Nombre_Nivel, lam, threshold)
    # Mayor 'lam' y 'threshold' = Mayor compresión, menor tamaño, menor calidad.
    compression_levels = [
        ("Nivel_1_AltaCalidad",  lam := 1.0,   thresh := 5.0),
        ("Nivel_2_MediaAlta",    lam := 10.0,  thresh := 15.0),
        ("Nivel_3_Media",        lam := 30.0,  thresh := 30.0),
        ("Nivel_4_Baja",         lam := 60.0,  thresh := 50.0),
        ("Nivel_5_MuyBaja",      lam := 100.0, thresh := 80.0),
    ]
    
    print("\nIniciando pruebas de compresión...")
    print("-" * 60)
    
    for level_name, lam, thresh in compression_levels:
        start_time = time.time()
        
        # --- COMPRESIÓN ---
        # Construye el quadtree, poda con RDO y asigna modos
        compressor.compress(
            image_rgb=img_rgb, 
            saliency_map=saliency_map, 
            threshold=thresh, 
            alpha=1.0, # Ajusta qué tanto la saliencia baja el umbral
            lam=lam, 
            beta=2.0   # Ajusta qué tanto la saliencia aumenta la importancia en RDO
        )
        
        # Empaqueta en un binario real con Zlib y headers
        pad_h, pad_w = compressor.root.h, compressor.root.w
        compressed_bytes = codec.compress(compressor.root, (pad_h, pad_w), compressor.dynamic_quality)
        
        # Calcular tamaño y BPP (Bits per Pixel)
        compressed_size = len(compressed_bytes)
        bpp = (compressed_size * 8) / (h * w)
        
        # --- DESCOMPRESIÓN ---
        # Parseo del binario
        dec_root, dec_shape, dec_quality = codec.decompress(compressed_bytes)
        
        # Restaurar variables del compresor para reconstruir (necesita las medidas originales para hacer el recorte del padding)
        compressor.orig_h = h
        compressor.orig_w = w
        compressor.dynamic_quality = dec_quality
        
        # Reconstruir los pixeles a partir del árbol decodificado
        rec_img_rgb = compressor.reconstruct(dec_root, override_quality=dec_quality)
        rec_img_bgr = cv2.cvtColor(rec_img_rgb, cv2.COLOR_RGB2BGR)
        
        # --- MÉTRICAS ---
        # Calcular PSNR, SSIM, etc.
        results = metrics.calculate_all(img_rgb, rec_img_rgb, saliency_map)
        psnr = results.get('psnr', 0)
        ssim = results.get('ssim', 0)
        sw_ssim = results.get('sw_ssim', 0) # Saliency-Weighted SSIM
        
        elapsed = time.time() - start_time
        
        # --- GUARDAR RESULTADOS ---
        # Guardar imagen reconstruida
        out_filename = f"{level_name}_bpp_{bpp:.3f}_psnr_{psnr:.2f}.png"
        cv2.imwrite(os.path.join(output_dir, out_filename), rec_img_bgr)
        
        # Imprimir reporte del nivel
        print(f"Nivel: {level_name} (lam={lam}, th={thresh})")
        print(f"  Tamaño Binario : {compressed_size / 1024:.2f} KB")
        print(f"  BPP (Bits/Pix) : {bpp:.4f} bpp")
        print(f"  PSNR           : {psnr:.2f} dB")
        print(f"  SSIM           : {ssim:.4f}")
        print(f"  SW-SSIM        : {sw_ssim:.4f}")
        print(f"  Tiempo total   : {elapsed:.2f} s")
        print("-" * 60)

if __name__ == "__main__":
    # Cambia 'imagen_prueba.jpg' por la ruta de tu imagen de testeo
    # y 'resultados_tesis' por la carpeta donde quieres guardar el experimento.
    test_image = "data/kodak/kodim04.png" 
    
    # Crear una imagen falsa de prueba por si la corres directamente sin imagen local
    if not os.path.exists(test_image):
        print("Creando imagen de prueba dummy...")
        dummy = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        cv2.circle(dummy, (256, 256), 100, (0, 0, 255), -1) # Círculo rojo como zona "saliency"
        cv2.imwrite(test_image, dummy)

    run_compression_experiment(test_image, "./resultados_tesis")