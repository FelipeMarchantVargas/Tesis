import argparse
import cv2
import numpy as np
import pickle
import zlib
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Importaciones locales (Asumiendo estructura src/)
from src.saliency import SaliencyDetector
from src.quadtree import QuadtreeCompressor, QuadtreeNode
from src.codec import QuadtreeCodec

def save_compressed_data(filepath: str, compressor_instance, shape):
    """Guarda usando el QuadtreeCodec (Binario optimizado)."""
    try:
        codec = QuadtreeCodec()
        # Aquí ocurre la magia: Árbol -> Bytes -> Zlib
        compressed_bytes = codec.compress(compressor_instance.root, shape)
        
        with open(filepath, 'wb') as f:
            f.write(compressed_bytes)
            
        print(f"[IO] Guardado optimizado en {filepath}. Tamaño: {len(compressed_bytes)/1024:.2f} KB")
    except Exception as e:
        print(f"[Error] Fallo al guardar archivo: {e}")
        sys.exit(1)

def load_compressed_data(filepath: str):
    """Carga y reconstruye el árbol desde bytes."""
    try:
        with open(filepath, 'rb') as f:
            compressed_bytes = f.read()
            
        codec = QuadtreeCodec()
        root, shape = codec.decompress(compressed_bytes)
        
        return root, shape
    except Exception as e:
        print(f"[Error] Fallo al cargar archivo {filepath}: {e}")
        sys.exit(1)

def handle_compress(args):
    """Flujo de Compresión."""
    # 1. Cargar Imagen
    img_path = Path(args.input)
    if not img_path.exists():
        print(f"[Error] La imagen {args.input} no existe.")
        sys.exit(1)

    print(f"--- Iniciando Compresión: {img_path.name} ---")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("[Error] No se pudo decodificar la imagen.")
        sys.exit(1)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 2. Generar Mapa de Saliencia
    print("[1/4] Generando Mapa de Saliencia...")
    # Usar modo 'cpu' o 'cuda' según disponibilidad, o forzar con args si quisieras extenderlo
    detector = SaliencyDetector(weights_path=args.model)
    saliency_map = detector.get_saliency_map(img_rgb)

    # 3. Compresión Quadtree
    print(f"[2/4] Construyendo Quadtree (Th={args.threshold}, Alpha={args.alpha})...")
    compressor = QuadtreeCompressor(min_block_size=4, max_depth=12)
    
    start_time = time.time()
    compressor.compress(img_rgb, saliency_map, args.threshold, args.alpha)
    
    # El balanceo ya está integrado dentro de compress() en la versión v2, 
    # pero si lo separaste, asegúrate de llamarlo aquí.
    # En la última versión que te di, compress() llama a balance_tree() internamente.
    
    duration = time.time() - start_time
    leaf_count = len(compressor.leaves)
    print(f"[3/4] Compresión finalizada en {duration:.4f}s. Nodos Hoja: {leaf_count}")

    # 4. Guardar
    print("[4/4] Guardando archivo comprimido...")
    payload = {
        'shape': (h, w),
        'leaves': compressor.leaves,
        'metadata': {
            'threshold': args.threshold,
            'alpha': args.alpha,
            'original_size': img_rgb.nbytes,
            'timestamp': time.time()
        }
    }
    save_compressed_data(args.output, payload)
    print("--- Proceso Completado ---")

def handle_reconstruct(args):
    """Flujo de Reconstrucción."""
    print(f"--- Iniciando Reconstrucción: {args.input} ---")
    
    # Cargamos usando el nuevo codec
    root, shape = load_compressed_data(args.input)
    print(f"[1/2] Datos decodificados. Dimensiones: {shape}")

    # Preparamos compresor para reconstruir
    compressor = QuadtreeCompressor()
    compressor.root = root 
    # Importante: Necesitamos reconstruir la lista de hojas (leaves) para el renderizado
    # porque el codec solo recuperó la estructura de árbol.
    compressor.leaves = []
    compressor._collect_leaves_recursive(compressor.root) # Usamos el método existente para llenar la lista
    
    start_time = time.time()
    rec_rgb = compressor.reconstruct(shape)
    duration = time.time() - start_time
    
    print(f"[2/2] Imagen reconstruida en {duration:.4f}s.")

    # 3. Guardar Imagen
    rec_bgr = cv2.cvtColor(rec_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, rec_bgr)
    print(f"[IO] Imagen guardada en {args.output}")

def handle_visualize(args):
    """Flujo de Visualización (Wireframe)."""
    # Este comando corre el pipeline de compresión pero guarda el wireframe en lugar del pickle
    img_path = Path(args.input)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pipeline
    detector = SaliencyDetector(weights_path=args.model)
    saliency_map = detector.get_saliency_map(img_rgb)

    compressor = QuadtreeCompressor(min_block_size=4, max_depth=12)
    compressor.compress(img_rgb, saliency_map, args.threshold, args.alpha)

    # Visualización
    print("[Vis] Generando Wireframe...")
    wireframe = compressor.visualize_structure(img_rgb.shape[:2])
    
    # Superposición: Imagen reconstruida + Bordes verdes
    rec_rgb = compressor.reconstruct(img_rgb.shape[:2])
    
    # Blend simple: 80% imagen, 100% líneas (donde no sean negras)
    mask = np.all(wireframe == 0, axis=2) # Máscara de fondo negro
    final_vis = rec_rgb.copy()
    final_vis[~mask] = wireframe[~mask] # Pintar líneas sobre la imagen
    
    out_bgr = cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)
    print(f"[IO] Visualización guardada en {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Tesis: Compresión de Imágenes Perceptual mediante Quadtrees.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Comando: Compress ---
    parser_comp = subparsers.add_parser('compress', help='Comprime una imagen a un archivo .pkl comprimido.')
    parser_comp.add_argument('--input', '-i', required=True, type=str, help='Ruta a la imagen de entrada.')
    parser_comp.add_argument('--output', '-o', required=True, type=str, help='Ruta de salida archivo .pkl.')
    parser_comp.add_argument('--model', '-m', type=str, default=None, help='Ruta a los pesos U2-Net (.pth). Opcional.')
    parser_comp.add_argument('--threshold', '-t', type=float, default=10.0, help='Umbral de error (MSE/StdDev).')
    parser_comp.add_argument('--alpha', '-a', type=float, default=0.5, help='Factor de importancia semántica (0.0 - 1.0).')
    parser_comp.set_defaults(func=handle_compress)

    # --- Comando: Reconstruct ---
    parser_rec = subparsers.add_parser('reconstruct', help='Reconstruye una imagen desde un archivo .pkl.')
    parser_rec.add_argument('--input', '-i', required=True, type=str, help='Ruta al archivo .pkl.')
    parser_rec.add_argument('--output', '-o', required=True, type=str, help='Ruta de salida imagen .png/.jpg.')
    parser_rec.set_defaults(func=handle_reconstruct)

    # --- Comando: Visualize ---
    parser_vis = subparsers.add_parser('visualize', help='Genera una imagen con la estructura Quadtree superpuesta.')
    parser_vis.add_argument('--input', '-i', required=True, type=str, help='Ruta a la imagen de entrada.')
    parser_vis.add_argument('--output', '-o', required=True, type=str, help='Ruta de salida imagen .png.')
    parser_vis.add_argument('--model', '-m', type=str, default=None, help='Ruta a pesos U2-Net.')
    parser_vis.add_argument('--threshold', '-t', type=float, default=10.0, help='Umbral de error.')
    parser_vis.add_argument('--alpha', '-a', type=float, default=0.5, help='Factor alpha.')
    parser_vis.set_defaults(func=handle_visualize)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()