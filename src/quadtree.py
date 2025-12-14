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

    def compress(self, image_rgb: np.ndarray, saliency_map: np.ndarray, threshold: float, alpha: float, lam: float = 10.0, beta: float = 2.0):
        """
        Flujo Híbrido: Top-Down (CNN) + Bottom-Up (RDO Saliency).
        
        Nuevos Params:
            lam: Lambda para RDO (Controla bitrate final).
            beta: Peso de la saliencia en RDO (Protección de tesis).
        """
        self._img = image_rgb
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        self._integral_img = IntegralImageWrapper(gray)
        
        saliency_uint8 = (saliency_map * 255).astype(np.uint8)
        self._integral_saliency = IntegralImageWrapper(saliency_uint8)

        h, w = image_rgb.shape[:2]
        self.root = QuadtreeNode(0, 0, h, w, 0)
        
        print("[Quadtree] 1. División Top-Down (Guiada por CNN)...")
        # Usamos un umbral inicial RELAJADO (más bajo) para sobre-segmentar un poco
        # y dejar que el RDO tome la decisión final de limpieza.
        initial_threshold = threshold * 0.8 
        self._recursive_split(self.root, initial_threshold, alpha)

        print("[Quadtree] 2. Balanceo Geométrico...")
        self.balance_tree()

        # AQUÍ ESTÁ LA MAGIA DE TU TESIS
        print(f"[Quadtree] 3. Optimización RDO (Lambda={lam}, Saliencia={beta})...")
        # Primero aseguramos que todos los nodos tengan datos para calcular distorsión
        # (Aunque _prune lo calcula on-the-fly, es bueno tener la estructura lista)
        self.leaves = [] 
        # Nota: No necesitamos collect_leaves aquí porque prune recorre desde root,
        # pero RDO necesita leer píxeles, self._img ya está seteado.
        
        self.prune_with_saliency_rdo(lam=lam, beta=beta)

        print(f"[Quadtree] Finalizado. Hojas finales: {len(self.leaves)}")
        
        # Limpieza de referencias pesadas
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

    def reconstruct(self, node: QuadtreeNode) -> np.ndarray:
        """
        Reconstruye la imagen usando Interpolación Bilineal desde un nodo dado.
        Toma las dimensiones directamente del nodo raíz.
        """
        h, w = node.h, node.w
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        
        # Llamamos al helper recursivo
        self._reconstruct_recursive_interpolated(node, canvas)
        
        return np.clip(canvas, 0, 255).astype(np.uint8)

    def _reconstruct_recursive_interpolated(self, node, canvas):
        """Recorre el árbol y pinta las hojas usando interpolación."""
        if not node.is_leaf:
            for child in node.children:
                self._reconstruct_recursive_interpolated(child, canvas)
        else:
            # --- Lógica de Pintado de Hoja (Interpolación) ---
            y, x, h, w = node.y, node.x, node.h, node.w
            
            # 1. Obtener colores (asegurando que no sean None)
            c_tl = node.color_tl if node.color_tl is not None else np.zeros(3)
            c_tr = node.color_tr if node.color_tr is not None else np.zeros(3)
            c_bl = node.color_bl if node.color_bl is not None else np.zeros(3)
            c_br = node.color_br if node.color_br is not None else np.zeros(3)

            # 2. Obtener mallas de interpolación (usando tu cache existente)
            xv, yv = self._get_interpolation_grids(h, w)
            
            # 3. Calcular gradiente bilineal
            top = c_tl * (1 - xv) + c_tr * xv
            bottom = c_bl * (1 - xv) + c_br * xv
            block = top * (1 - yv) + bottom * yv
            
            # 4. Pintar en el canvas
            canvas[y:y+h, x:x+w] = block
    # --- MÉTODOS DE OPTIMIZACIÓN RDO (SALIENCY-AWARE) ---

    def prune_with_saliency_rdo(self, lam: float, beta: float = 1.0):
        """
        Poda el árbol usando optimización Lagrangiana ponderada por Saliencia.
        Objetivo: J = (Distortion * Importance) + (Lambda * Rate)
        
        Args:
            lam (float): Penalización por bits (Lambda). Mayor = Más compresión.
                         Rango sugerido: 1.0 a 50.0.
            beta (float): Peso de la tesis. Controla cuánto protege la saliencia a un bloque.
                          0.0 = RDO estándar (ignora saliencia).
                          1.0 = Saliencia normal.
                          >1.0 = Protección agresiva de zonas salientes.
        """
        if self.root is None: return

        print(f"[RDO] Optimizando estructura (Lambda={lam}, Beta={beta})...")
        self._prune_saliency_recursive(self.root, lam, beta)
        
        # Actualizar la lista oficial de hojas después de la poda
        self.leaves = []
        self._collect_leaves_recursive(self.root)

    def _prune_saliency_recursive(self, node: QuadtreeNode, lam: float, beta: float) -> float:
        """Retorna el costo mínimo (J) del subárbol."""
        
        # 1. Costo de Bit Rate (Estimación para modo 'optimized')
        # - Nodo Hoja: 1 bit estructura + 48 bits color (6 bytes) = 49 bits
        # - Nodo Split: 1 bit estructura
        r_leaf = 49.0
        r_split = 1.0

        # 2. Factor de Importancia (Tu Tesis)
        # Obtenemos la saliencia promedio del bloque en O(1)
        mean_sal, _ = self._integral_saliency.get_stats(node.y, node.x, node.h, node.w)
        norm_sal = mean_sal / 255.0  # [0.0 - 1.0]
        
        # Si beta=0, weight=1 (RDO clásico). Si beta=1 y saliencia=1, weight=2 (El error duele el doble).
        importance_weight = 1.0 + (beta * norm_sal)

        # 3. Calcular Costo de ser Hoja (J_leaf)
        # Calculamos el error real (SSD) de interpolar este bloque
        distortion = self._calculate_distortion(node)
        
        # Costo Hoja = (Error * Importancia) + (Lambda * Bits)
        j_leaf = (distortion * importance_weight) + (lam * r_leaf)

        if node.is_leaf:
            return j_leaf

        # 4. Calcular Costo de ser Split (J_split)
        # Costo Split = 1 bit + Suma de costos óptimos de los hijos
        j_children = 0.0
        for child in node.children:
            j_children += self._prune_saliency_recursive(child, lam, beta)
            
        j_split = r_split * lam + j_children # El bit del split también se multiplica por lambda? No, lambda multiplica al Rate total.
        # Corrección fórmula lagrangiana estándar: J = D + lambda * R
        # R_split_total = 1 (flag) + bits_hijos
        # Entonces: J_split = (Sum_D_hijos * Importancia) + lambda * (1 + Sum_R_hijos)
        # Simplificando la recursión: la función retorna J, que ya incluye lambda*R.
        # Solo sumamos el costo del bit de estructura actual:
        j_split = (lam * 1.0) + j_children

        # 5. La Decisión (Poda)
        if j_leaf <= j_split:
            # Es "más barato" (o más eficiente) ser hoja. Podamos.
            node.children = [] 
            # IMPORTANTE: Al convertir en hoja, debemos asegurar que tenga datos de color
            if node.color_tl is None:
                self._capture_leaf_data(node)
            return j_leaf
        else:
            # Vale la pena mantener la división
            return j_split

    def _calculate_distortion(self, node: QuadtreeNode) -> float:
        """Calcula el Error Cuadrático (SSD) entre la imagen original y la interpolación."""
        y, x, h, w = node.y, node.x, node.h, node.w
        
        # Extraer patch original
        original = self._img[y:y+h, x:x+w].astype(np.float32)

        # Simular interpolación (usando esquinas reales)
        # Nota: Usamos las esquinas de la imagen original para simular el mejor caso de reconstrucción
        h_img, w_img = self._img.shape[:2]
        
        # Coordenadas seguras
        y1, x1 = min(y + h - 1, h_img - 1), min(x + w - 1, w_img - 1)
        
        c_tl = self._img[y, x].astype(np.float32)
        c_tr = self._img[y, x1].astype(np.float32)
        c_bl = self._img[y1, x].astype(np.float32)
        c_br = self._img[y1, x1].astype(np.float32)

        # Obtener grids cacheados
        xv, yv = self._get_interpolation_grids(h, w)
        
        # Interpolar
        top = c_tl * (1 - xv) + c_tr * xv
        bottom = c_bl * (1 - xv) + c_br * xv
        reconstructed = top * (1 - yv) + bottom * yv
        
        # Calcular SSD (Sum of Squared Differences)
        diff = original - reconstructed
        ssd = np.sum(diff ** 2)
        return ssd
    
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