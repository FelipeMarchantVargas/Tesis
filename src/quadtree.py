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

    def __init__(self, min_block_size: int = 4, max_depth: int = 12):
        self.min_block_size = min_block_size
        self.max_depth = max_depth
        self.root: Optional[QuadtreeNode] = None
        self.leaves: List[QuadtreeNode] = []
        
        # Referencias temporales
        self._img: Optional[np.ndarray] = None
        self._integral_img: Optional[IntegralImageWrapper] = None
        self._integral_saliency: Optional[IntegralImageWrapper] = None

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

        effective_threshold = threshold * (1.0 - (alpha * saliency_mean))

        effective_threshold = max(effective_threshold, 1.0)

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
        Asegura la restricción de nivel.
        Itera hasta que no haya violaciones de la regla: |depth_node - depth_neighbor| <= 1.
        """
        balanced = False
        pass_count = 0
        
        while not balanced:
            balanced = True
            pass_count += 1
            # Recolectamos hojas actuales (esto cambia dinámicamente si dividimos)
            current_leaves = []
            self._collect_leaves_no_data(self.root, current_leaves)
            
            for leaf in current_leaves:
                # Puntos de chequeo en los 4 vecinos (midpoints de los bordes)
                # Usamos coordenadas ligeramente fuera del nodo
                mid_y = leaf.y + leaf.h // 2
                mid_x = leaf.x + leaf.w // 2
                
                neighbor_points = [
                    (leaf.y - 1, mid_x),          # Norte
                    (leaf.y + leaf.h, mid_x),     # Sur
                    (mid_y, leaf.x - 1),          # Oeste
                    (mid_y, leaf.x + leaf.w)      # Este
                ]

                for ny, nx in neighbor_points:
                    neighbor = self._get_leaf_at(ny, nx)
                    
                    if neighbor is None:
                        continue # Borde de la imagen

                    # La Regla:
                    # Si mi vecino es mucho MENOS profundo que yo (más grande),
                    # y la diferencia es > 1, el vecino debe dividirse.
                    # (Nota: Si yo soy el menos profundo, eventualmente se iterará sobre mi vecino
                    # y él me detectará a mí. Solo necesitamos manejar un sentido de la desigualdad).
                    
                    if leaf.depth > neighbor.depth + 1:
                        # El vecino es demasiado grande comparado conmigo. Dividirlo.
                        self._split_node(neighbor)
                        balanced = False 
                        # IMPORTANTE: Al dividir, invalidamos la lista `current_leaves`.
                        # Rompemos el bucle interno para reiniciar la recolección de hojas.
                        break 
                
                if not balanced:
                    break
            
            if pass_count > 50: # Safety break
                print("Advertencia: Límite de pases de balanceo alcanzado.")
                break

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
        """Reconstrucción vectorizada (Igual que antes)."""
        canvas = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.float32)

        for node in self.leaves:
            if node.color_tl is None: continue 

            x_range = np.linspace(0, 1, node.w).astype(np.float32)
            y_range = np.linspace(0, 1, node.h).astype(np.float32)
            xv, yv = np.meshgrid(x_range, y_range)
            xv = xv[..., np.newaxis] 
            yv = yv[..., np.newaxis]

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