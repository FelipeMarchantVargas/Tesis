import cv2
import numpy as np
from typing import Tuple, List, Optional

# ==============================================================================
# 1. La Estructura de Datos del Nodo
# ==============================================================================
class QuadtreeNode:
    """
    Representa un nodo en el Quadtree. Un nodo puede ser una rama (con 4 hijos)
    o una hoja (sin hijos, con un color definido).
    """
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Una lista de 4 hijos. Si está vacía, es un nodo hoja.
        self.children: List['QuadtreeNode'] = []
        
        # El color promedio de la región si el nodo es una hoja.
        # Se almacena como (B, G, R) para compatibilidad con OpenCV.
        self.color: Optional[Tuple[float, float, float]] = None

    @property
    def is_leaf(self) -> bool:
        """Retorna True si el nodo es una hoja (no tiene hijos)."""
        return not self.children

# ==============================================================================
# 2. El Compresor Quadtree
# ==============================================================================
class QuadtreeCompressor:
    """
    Implementa el algoritmo de compresión y reconstrucción de imágenes
    basado en un Quadtree tradicional. La subdivisión se basa en la
    desviación estándar del color de los píxeles.
    """
    def __init__(self, threshold: float = 15.0, max_depth: int = 8, min_size: int = 4):
        """
        Inicializa el compresor con los parámetros de compresión.

        Args:
            threshold (float): El umbral de error (desviación estándar). Regiones
                               con un error menor a este valor no se subdividen.
                               Valores más altos = más compresión.
            max_depth (int): La profundidad máxima del árbol. Limita la subdivisión
                             en áreas de alto detalle.
            min_size (int): El tamaño mínimo de un cuadrante para ser subdividido.
        """
        self.threshold = threshold
        self.max_depth = max_depth
        self.min_size = min_size
        self.root: Optional[QuadtreeNode] = None
        self._leaf_nodes = []

    def _calculate_error(self, image_region: np.ndarray) -> float:
        """
        Calcula el error de una región. Basado en papers, una métrica robusta
        es la desviación estándar media de los canales de color.
        """
        if image_region.size == 0:
            return 0.0
        # Calcula la desviación estándar para cada canal (B, G, R) y luego promedia.
        b, g, r = cv2.split(image_region)
        std_dev = (np.std(b) + np.std(g) + np.std(r)) / 3
        return std_dev

    def _calculate_average_color(self, image_region: np.ndarray) -> Tuple[float, float, float]:
        """Calcula el color promedio de una región."""
        # mean() sobre los ejes de pixeles (0 y 1) devuelve el promedio para cada canal.
        avg_color = np.mean(image_region, axis=(0, 1))
        return (avg_color[0], avg_color[1], avg_color[2]) # (B, G, R)

    def _subdivide(self, node: QuadtreeNode, image: np.ndarray, current_depth: int):
        """
        Función recursiva principal. Decide si subdividir un nodo o convertirlo en hoja.
        """
        # --- Condiciones de Parada ---
        if (current_depth >= self.max_depth or 
            node.width <= self.min_size or 
            node.height <= self.min_size):
            node.color = self._calculate_average_color(image)
            self._leaf_nodes.append(node)
            return

        # --- Criterio de Subdivisión ---
        error = self._calculate_error(image)
        if error < self.threshold:
            node.color = self._calculate_average_color(image)
            self._leaf_nodes.append(node)
            return

        # --- Lógica de Subdivisión ---
        # Si ninguna condición de parada se cumple, dividimos el nodo.
        mid_x = node.x + node.width // 2
        mid_y = node.y + node.height // 2
        
        # Coordenadas y dimensiones de los 4 hijos
        # (Noroeste, Noreste, Suroeste, Sureste)
        children_coords = [
            (node.x, node.y, node.width // 2, node.height // 2),  # NW
            (mid_x, node.y, node.width - node.width // 2, node.height // 2),  # NE
            (node.x, mid_y, node.width // 2, node.height - node.height // 2),  # SW
            (mid_x, mid_y, node.width - node.width // 2, node.height - node.height // 2)  # SE
        ]
        
        for x, y, w, h in children_coords:
            child_node = QuadtreeNode(x, y, w, h)
            node.children.append(child_node)
            
            # Recortamos la porción de la imagen correspondiente al hijo
            child_image_region = image[y - node.y : y - node.y + h, x - node.x : x - node.x + w]
            
            # Llamada recursiva
            self._subdivide(child_node, child_image_region, current_depth + 1)

    def compress(self, image: np.ndarray):
        """
        Comprime la imagen dada construyendo el Quadtree.
        """
        self._leaf_nodes = [] # Reiniciar la lista de hojas
        height, width, _ = image.shape
        self.root = QuadtreeNode(0, 0, width, height)
        self._subdivide(self.root, image, 0)
        print(f"Compresión completada. Número de nodos hoja: {len(self._leaf_nodes)}")

    def reconstruct(self) -> Optional[np.ndarray]:
        """
        Reconstruye la imagen a partir de los nodos hoja del Quadtree.
        """
        if not self.root:
            print("Error: El árbol está vacío. Ejecuta compress() primero.")
            return None
            
        # Crea un lienzo negro del tamaño de la imagen original
        image_shape = (self.root.height, self.root.width, 3)
        reconstructed_image = np.zeros(image_shape, dtype=np.uint8)
        
        # Dibuja un rectángulo para cada hoja
        for leaf in self._leaf_nodes:
            # Convierte el color promedio a valores enteros de 8 bits
            color_bgr = tuple(int(c) for c in leaf.color)
            
            top_left = (leaf.x, leaf.y)
            bottom_right = (leaf.x + leaf.width, leaf.y + leaf.height)
            
            cv2.rectangle(reconstructed_image, top_left, bottom_right, color_bgr, -1) # -1 para rellenar
            
        return reconstructed_image

# ==============================================================================
# 3. Bloque de Ejecución para Prueba
# ==============================================================================
if __name__ == '__main__':
    # --- Carga y Configuración ---
    image_path = 'data/test_image.jpg'
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en: {image_path}")
    except FileNotFoundError as e:
        print(e)
        exit()

    h, w, _ = original_image.shape
    scale = 512 / max(h, w)
    small_image = cv2.resize(original_image, (int(w*scale), int(h*scale)))

    # --- Compresión y Reconstrucción ---
    compressor = QuadtreeCompressor(threshold=0.1, max_depth=15)
    
    print("Comprimiendo la imagen...")
    compressor.compress(small_image)
    
    print("Reconstruyendo la imagen...")
    reconstructed_image = compressor.reconstruct()

    # --- Visualización y Guardado usando solo OpenCV y NumPy ---
    if reconstructed_image is not None:
        # Poner las dos imágenes una al lado de la otra
        # hstack requiere que las imágenes tengan la misma altura, lo cual es nuestro caso
        comparison = np.hstack([small_image, reconstructed_image])

        # Añadir texto a la imagen para clarificar
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison, f"Reconstruida ({len(compressor._leaf_nodes)} hojas)", (small_image.shape[1] + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Guardamos la imagen de comparación en la carpeta 'results'
        output_path = 'results/comparison_opencv.png'
        cv2.imwrite(output_path, comparison)
        
        print(f"¡Éxito! Tu lógica de compresión funcionó.")
        print(f"La imagen de comparación ha sido guardada en: {output_path}")