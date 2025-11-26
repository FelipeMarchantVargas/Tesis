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

    def _subdivide(self, node: QuadtreeNode, image: np.ndarray, 
                   saliency_map: Optional[np.ndarray], current_depth: int):
        """
        Función recursiva que decide si dividir un nodo basándose en el error de color
        y la importancia perceptual (Saliency Map).
        """
        
        # 1. Condiciones de Parada (Hard Limits)
        # Si alcanzamos la profundidad máxima o el tamaño mínimo, nos detenemos.
        if (current_depth >= self.max_depth or 
            node.width <= self.min_size or 
            node.height <= self.min_size):
            node.color = self._calculate_average_color(image)
            self._leaf_nodes.append(node)
            return

        # 2. Cálculo de Métricas
        # A. Error estadístico (Desviación estándar del color)
        std_dev_error = self._calculate_error(image)
        
        # B. Factor de Importancia (Semántico)
        importance_factor = 0.0
        if saliency_map is not None and saliency_map.size > 0:
            # Calculamos el promedio de prominencia en esta región (0.0 a 1.0)
            importance_factor = np.mean(saliency_map)

        # 3. Cálculo del Umbral Dinámico (Lógica Híbrida)
        # Definimos 'alpha' aquí (o podrías pasarlo en __init__).
        # alpha = 0.8 significa que la IA tiene un 80% de influencia en reducir el umbral.
        alpha = 0.8 
        
        # Fórmula: Si importance_factor es alto (1.0), el umbral efectivo se reduce.
        # Umbral bajo = Mayor sensibilidad = Más subdivisiones = Más calidad.
        effective_threshold = self.threshold * (1.0 - (alpha * importance_factor))

        # 4. Decisión de Subdivisión
        # Si el error actual es menor que nuestro umbral ajustado, consideramos
        # que la región es suficientemente homogénea y no dividimos más.
        if std_dev_error < effective_threshold:
            node.color = self._calculate_average_color(image)
            self._leaf_nodes.append(node)
            return

        # 5. Ejecución de la Subdivisión (Recursión)
        # Calculamos dimensiones locales para cortar los arrays
        h_local, w_local = image.shape[:2]
        half_w = w_local // 2
        half_h = h_local // 2
        
        # Definimos puntos medios absolutos para las coordenadas de los nodos hijos
        mid_x = node.x + half_w
        mid_y = node.y + half_h
        
        # Configuración de los 4 hijos: (x_abs, y_abs, w, h)
        children_coords = [
            (node.x, node.y, half_w, half_h),           # NW (Noroeste)
            (mid_x, node.y, w_local - half_w, half_h),  # NE (Noreste)
            (node.x, mid_y, half_w, h_local - half_h),  # SW (Suroeste)
            (mid_x, mid_y, w_local - half_w, h_local - half_h) # SE (Sureste)
        ]
        
        # Definimos los cortes (slices) para los arrays numpy (image y saliency)
        # Corresponden a: [slice_y, slice_x]
        slices = [
            (slice(0, half_h), slice(0, half_w)),              # NW
            (slice(0, half_h), slice(half_w, w_local)),        # NE
            (slice(half_h, h_local), slice(0, half_w)),        # SW
            (slice(half_h, h_local), slice(half_w, w_local))   # SE
        ]

        for (nx, ny, nw, nh), (sl_y, sl_x) in zip(children_coords, slices):
            # Crear nodo hijo
            child_node = QuadtreeNode(nx, ny, nw, nh)
            node.children.append(child_node)
            
            # Recortar imagen para el hijo
            child_image = image[sl_y, sl_x]
            
            # Recortar mapa de prominencia para el hijo (si existe)
            child_saliency = None
            if saliency_map is not None:
                child_saliency = saliency_map[sl_y, sl_x]
            
            # Llamada recursiva pasando los recortes
            self._subdivide(child_node, child_image, child_saliency, current_depth + 1)

    def compress(self, image: np.ndarray, saliency_map: Optional[np.ndarray] = None):
        """
        Comprime la imagen dada construyendo el Quadtree.
        Acepta un mapa de prominencia opcional.
        """
        self._leaf_nodes = [] # Reiniciar la lista de hojas
        height, width, _ = image.shape
        self.root = QuadtreeNode(0, 0, width, height)
        
        # CORRECCIÓN AQUÍ:
        # Pasamos el saliency_map (que puede ser None) a la función recursiva.
        self._subdivide(self.root, image, saliency_map, 0)
        
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
    
    def visualize_structure(self, background_image: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Dibuja los bordes de los nodos hoja para visualizar la densidad del Quadtree.
        Si se pasa una imagen de fondo, dibuja sobre ella. Si no, usa un fondo negro.
        """
        if not self.root:
            print("Error: El árbol está vacío. Ejecuta compress() primero.")
            return None

        # 1. Preparar el lienzo (Canvas)
        height, width = self.root.height, self.root.width
        
        if background_image is not None:
            # Si nos dan una imagen, trabajamos sobre una copia para no alterar la original
            canvas = background_image.copy()
            # Usaremos color verde brillante para que resalte sobre la foto
            line_color = (0, 255, 0) 
        else:
            # Si no, creamos un fondo negro
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            # Usaremos color blanco o verde
            line_color = (0, 255, 0) 

        # 2. Dibujar los rectángulos
        for leaf in self._leaf_nodes:
            top_left = (leaf.x, leaf.y)
            bottom_right = (leaf.x + leaf.width, leaf.y + leaf.height)
            
            # El argumento 'thickness=1' dibuja solo el borde.
            cv2.rectangle(canvas, top_left, bottom_right, line_color, 1)

        return canvas

# ==============================================================================
# 3. Bloque de Ejecución para Prueba
# ==============================================================================
if __name__ == '__main__':
    # --- 1. Carga y Configuración ---
    image_path = 'data/test_image_1.jpg' # Asegúrate que esta ruta exista o cambia el nombre
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en: {image_path}")
    except FileNotFoundError as e:
        print(e)
        exit()

    # Redimensionar para pruebas rápidas
    h, w, _ = original_image.shape
    scale = 512 / max(h, w)
    small_image = cv2.resize(original_image, (int(w*scale), int(h*scale)))
    
    # --- 2. GENERACIÓN DE MAPA DE PROMINENCIA MOCK (Simulación) ---
    # Creamos un círculo blanco difuso en el centro para simular que eso es "importante"
    # Esto probará si tu lógica híbrida realmente subdivide más ahí.
    h_small, w_small, _ = small_image.shape
    saliency_mock = np.zeros((h_small, w_small), dtype=np.float32)
    
    center_x, center_y = w_small // 2, h_small // 2
    Y, X = np.ogrid[:h_small, :w_small]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Gradiente: 1.0 en el centro, bajando a 0.0 en los bordes
    radius = min(h_small, w_small) // 2
    saliency_mock = 1.0 - np.clip(dist_from_center / radius, 0, 1)
    
    # Visualizar el mapa de saliencia (opcional, para que veas qué le pasamos)
    cv2.imwrite('results/debug_saliency_mock.png', (saliency_mock * 255).astype(np.uint8))

    # --- 3. Compresión ---
    # Ajusta threshold y alpha en tu código para ver el efecto.
    # Un threshold alto (ej. 30) normalmente comprimiría mucho, 
    # pero el mapa de saliencia debería forzar detalle en el centro.
    compressor = QuadtreeCompressor(threshold=20, max_depth=8, min_size=4)
    
    print("Comprimiendo con guía de prominencia simulada...")
    # Pasamos el mapa simulado aquí
    compressor.compress(small_image, saliency_map=saliency_mock) 
    
    # --- 4. Visualizaciones ---
    print("Generando visualización de estructura...")
    
    # A) Ver la malla sobre fondo negro
    structure_viz = compressor.visualize_structure()
    cv2.imwrite('results/debug_structure_black.png', structure_viz)
    
    # B) Ver la malla sobre la imagen original (Muy útil)
    overlay_viz = compressor.visualize_structure(background_image=small_image)
    cv2.imwrite('results/debug_structure_overlay.png', overlay_viz)

    # C) Reconstrucción normal (Pintada)
    reconstructed_image = compressor.reconstruct()
    cv2.imwrite('results/output_reconstructed.png', reconstructed_image)

    print(f"¡Listo! Revisa la carpeta 'results/'.")
    print(f"- debug_saliency_mock.png: El mapa de guía.")
    print(f"- debug_structure_overlay.png: La malla verde sobre tu foto (Aquí deberías ver más densidad en el centro).")