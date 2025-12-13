import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class QuadtreeNode:
    """Nodo del Quadtree."""
    y: int
    x: int
    h: int
    w: int
    depth: int
    children: List['QuadtreeNode'] = field(default_factory=list)
    
    # Colores (RGB)
    color_tl: Optional[np.ndarray] = None
    color_tr: Optional[np.ndarray] = None
    color_bl: Optional[np.ndarray] = None
    color_br: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def contains(self, y: int, x: int) -> bool:
        """Verifica si una coordenada cae dentro de este nodo."""
        return (self.y <= y < self.y + self.h) and (self.x <= x < self.x + self.w)

class IntegralImageWrapper:
    """Helper estadístico O(1)."""
    def __init__(self, img: np.ndarray):
        self._sat = cv2.integral(img.astype(np.float64))
        self._sat_sq = cv2.integral(img.astype(np.float64) ** 2)

    def get_stats(self, y: int, x: int, h: int, w: int) -> Tuple[float, float]:
        y1, x1 = y + h, x + w
        area_sum = (self._sat[y1, x1] - self._sat[y, x1] - self._sat[y1, x] + self._sat[y, x])
        area_sq_sum = (self._sat_sq[y1, x1] - self._sat_sq[y, x1] - self._sat_sq[y1, x] + self._sat_sq[y, x])
        n = h * w
        if n == 0: return 0.0, 0.0
        mean = area_sum / n
        var = (area_sq_sum / n) - (mean ** 2)
        return mean, max(0.0, var)

class QuadtreeCompressor:
    """Compresor con soporte para Restricted Quadtree (Balanceado)."""

    _grid_cache = {}

    def __init__(self, min_block_size: int = 4, max_depth: int = 12):
        self.min_block_size = min_block_size
        self.max_depth = max_depth
        self.root: Optional[QuadtreeNode] = None
        self.leaves: List[QuadtreeNode] = []
        
        # Referencias temporales
        self._img: Optional[np.ndarray] = None
        self._integral_img: Optional[IntegralImageWrapper] = None
        self._integral_saliency: Optional[IntegralImageWrapper] = None
    
    def _get_interpolation_grids(self, h: int, w: int):
        """Devuelve grids cacheados para evitar meshgrid en cada hoja."""
        key = (h, w)
        if key not in self._grid_cache:
            x_range = np.linspace(0, 1, w, dtype=np.float32)
            y_range = np.linspace(0, 1, h, dtype=np.float32)
            xv, yv = np.meshgrid(x_range, y_range)
            self._grid_cache[key] = (xv[..., np.newaxis], yv[..., np.newaxis])
        return self._grid_cache[key]

    def compress(self, image_rgb: np.ndarray, saliency_map: np.ndarray, threshold: float, alpha: float):
        """Flujo principal: Construcción -> Balanceo -> Captura de Datos."""
        self._img = image_rgb
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        self._integral_img = IntegralImageWrapper(gray)
        
        saliency_uint8 = (saliency_map * 255).astype(np.uint8)
        self._integral_saliency = IntegralImageWrapper(saliency_uint8)

        h, w = image_rgb.shape[:2]
        self.root = QuadtreeNode(0, 0, h, w, 0)
        
        # 1. División Inicial Guiada por Semántica
        print("[Quadtree] Iniciando división recursiva...")
        self._recursive_split(self.root, threshold, alpha)

        # 2. Balanceo del Árbol (Restricted Quadtree)
        print("[Quadtree] Aplicando balanceo 2:1...")
        self.balance_tree()

        # 3. Finalización: Recolectar hojas finales y extraer colores
        print("[Quadtree] Capturando datos de color...")
        self.leaves = []
        self._collect_leaves_recursive(self.root)
        
        # Limpieza
        self._img = None
        self._integral_img = None
        self._integral_saliency = None

    def _recursive_split(self, node: QuadtreeNode, threshold: float, alpha: float):
        """División basada en error híbrido."""
        # Check dimensiones mínimas y profundidad máxima
        if (node.w <= self.min_block_size or node.h <= self.min_block_size or 
            node.depth >= self.max_depth):
            return

        # Cálculo de Error Híbrido
        _, variance = self._integral_img.get_stats(node.y, node.x, node.h, node.w)
        std_dev = np.sqrt(variance)
        
        mean_saliency_uint8, _ = self._integral_saliency.get_stats(node.y, node.x, node.h, node.w)
        saliency_mean = mean_saliency_uint8 / 255.0

        effective_threshold = threshold * np.exp(-alpha * saliency_mean)
        effective_threshold = max(effective_threshold, 1.0) # Evitar cero

        # Ahora comparamos el error real (std_dev) contra este umbral que se encogió
        if std_dev > effective_threshold:
            self._split_node(node)
            for child in node.children:
                self._recursive_split(child, threshold, alpha)

    def _split_node(self, node: QuadtreeNode):
        """Divide un nodo geométricamente."""
        half_w = node.w // 2
        half_h = node.h // 2
        y, x = node.y, node.x
        y_mid, x_mid = y + half_h, x + half_w
        
        node.children = [
            QuadtreeNode(y, x, half_h, half_w, node.depth + 1),         # TL
            QuadtreeNode(y, x_mid, half_h, node.w - half_w, node.depth + 1), # TR
            QuadtreeNode(y_mid, x, node.h - half_h, half_w, node.depth + 1), # BL
            QuadtreeNode(y_mid, x_mid, node.h - half_h, node.w - half_w, node.depth + 1) # BR
        ]

    def _get_leaf_at(self, y: int, x: int) -> Optional[QuadtreeNode]:
        """Busca la hoja que contiene la coordenada (y, x) descendiendo desde la raíz."""
        # Validación de límites de imagen
        if not self.root.contains(y, x):
            return None
            
        curr = self.root
        while not curr.is_leaf:
            # Determinar en qué hijo cae el punto
            found = False
            for child in curr.children:
                if child.contains(y, x):
                    curr = child
                    found = True
                    break
            if not found:
                return None # No debería ocurrir si la lógica es correcta
        return curr

    def balance_tree(self):
        """
        Balanceo 'Ripple' (Olas): Usa una cola para propagar divisiones.
        Solo re-verificamos nodos que han sido tocados o sus vecinos.
        """
        # 1. Recolectar todas las hojas iniciales
        process_queue = []
        self._collect_leaves_no_data(self.root, process_queue)
        
        # Convertimos la lista en un set para búsqueda rápida (opcional si usamos flags)
        # Pero para simplicidad, usaremos la cola directa.
        
        idx = 0
        while idx < len(process_queue):
            node = process_queue[idx]
            idx += 1
            
            if not node.is_leaf: continue # Si ya se dividió en el proceso, saltar

            # Buscar vecinos que violen la regla.
            # Nota: En quadtree de punteros sin "parent pointers", la búsqueda de vecinos
            # sigue siendo costosa. Si puedes agregar 'parent' a tu nodo, esto sería O(1).
            # Asumiremos la búsqueda top-down pero optimizada:
            
            # Estrategia: Chequear los 4 puntos medios externos
            neighbors = self._find_neighbors_of(node)
            
            for neighbor in neighbors:
                if neighbor.is_leaf and neighbor.depth < node.depth - 1:
                    # El vecino es muy grande (> 1 nivel de diferencia). Dividir.
                    self._split_node(neighbor)
                    
                    # Al dividir al vecino, sus hijos nuevos deben ser verificados
                    # para ver si ahora ellos violan reglas con SUS vecinos.
                    process_queue.extend(neighbor.children)

    def _find_neighbors_of(self, node: QuadtreeNode) -> List[QuadtreeNode]:
        """Encuentra los nodos hoja adyacentes (N, S, E, O)."""
        mid_y, mid_x = node.y + node.h // 2, node.x + node.w // 2
        points = [
            (node.y - 1, mid_x),          # Norte
            (node.y + node.h, mid_x),     # Sur
            (mid_y, node.x - 1),          # Oeste
            (mid_y, node.x + node.w)      # Este
        ]
        neighbors = []
        for y, x in points:
            n = self._get_leaf_at(y, x)
            if n: neighbors.append(n)
        return neighbors

    def _collect_leaves_no_data(self, node: QuadtreeNode, acc: List[QuadtreeNode]):
        """Auxiliar rápido para balanceo."""
        if node.is_leaf:
            acc.append(node)
        else:
            for child in node.children:
                self._collect_leaves_no_data(child, acc)

    def _collect_leaves_recursive(self, node: QuadtreeNode):
        """Recolección final que además captura los colores."""
        if node.is_leaf:
            self._capture_leaf_data(node)
            self.leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves_recursive(child)

    def _capture_leaf_data(self, node: QuadtreeNode):
        """Extrae colores de la imagen original."""
        h_img, w_img = self._img.shape[:2]
        y0, x0 = node.y, node.x
        y1, x1 = min(node.y + node.h - 1, h_img - 1), min(node.x + node.w - 1, w_img - 1)
        
        node.color_tl = self._img[y0, x0].astype(np.float32)
        node.color_tr = self._img[y0, x1].astype(np.float32)
        node.color_bl = self._img[y1, x0].astype(np.float32)
        node.color_br = self._img[y1, x1].astype(np.float32)

    def reconstruct(self, output_shape: Tuple[int, int]) -> np.ndarray:
        canvas = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.float32)

        for node in self.leaves:
            if node.color_tl is None: continue 
            
            # O(1) lookup
            xv, yv = self._get_interpolation_grids(node.h, node.w)

            # Vectorización pura
            top = node.color_tl * (1 - xv) + node.color_tr * xv
            bottom = node.color_bl * (1 - xv) + node.color_br * xv
            patch = top * (1 - yv) + bottom * yv
            
            canvas[node.y : node.y + node.h, node.x : node.x + node.w] = patch

        return np.clip(canvas, 0, 255).astype(np.uint8)
    
    def visualize_structure(self, image_shape: Tuple[int, int]) -> np.ndarray:
        vis = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        for node in self.leaves:
            cv2.rectangle(vis, (node.x, node.y), (node.x + node.w, node.y + node.h), (0, 255, 128), 1)
        return vis
    def reconstruct_blocks(self, output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstrucción clásica (estilo Minecraft/JPEG).
        Pinta todo el nodo con el color promedio de sus esquinas.
        """
        canvas = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)

        for node in self.leaves:
            if node.color_tl is None: continue 

            # Calculamos el color promedio del bloque
            # Promedio de las 4 esquinas
            avg_color = (node.color_tl + node.color_tr + node.color_bl + node.color_br) / 4.0
            
            # Pintamos el rectángulo sólido
            # (Clip para asegurar rango 0-255 y convertir a uint8)
            color_uint8 = np.clip(avg_color, 0, 255).astype(np.uint8)
            
            # Asignamos el color a toda la región del nodo
            canvas[node.y : node.y + node.h, node.x : node.x + node.w] = color_uint8

        return canvas