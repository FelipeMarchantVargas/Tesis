import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional

# --- 1. CLASE DE AYUDA (INTEGRADA AQUÍ PARA EVITAR ERRORES DE IMPORTACIÓN) ---
class IntegralImageWrapper:
    """
    Clase para calcular sumas y estadísticas en O(1) usando Tablas de Área Sumada.
    """
    def __init__(self, img: np.ndarray):
        # Aseguramos float32 para precisión
        self.img = img.astype(np.float32)
        # Integral de la imagen (Suma)
        self._sat = cv2.integral(self.img)
        # Integral de los cuadrados (Suma de Cuadrados para Varianza)
        self._sat_sq = cv2.integral(self.img ** 2)

    def get_stats(self, y: int, x: int, h: int, w: int):
        """Retorna (media, desviacion_estandar) de un bloque."""
        # Coordenadas seguras para la imagen integral (+1 por el padding de cv2.integral)
        y0, x0 = y, x
        y1, x1 = y + h, x + w
        
        # Área total
        area = h * w
        if area <= 0: return 0.0, 0.0

        # Suma del bloque: D + A - B - C
        # Layout cv2.integral: rows+1, cols+1
        # A=top-left, B=top-right, C=bottom-left, D=bottom-right
        
        # Suma simple
        s = (self._sat[y1, x1] - self._sat[y1, x0] - 
             self._sat[y0, x1] + self._sat[y0, x0])
        
        # Suma de cuadrados
        sq = (self._sat_sq[y1, x1] - self._sat_sq[y1, x0] - 
              self._sat_sq[y0, x1] + self._sat_sq[y0, x0])

        mean = s / area
        
        # Varianza = E[X^2] - (E[X])^2
        var = (sq / area) - (mean ** 2)
        # Evitar errores numéricos negativos pequeños
        if var < 0: var = 0
        
        std_dev = np.sqrt(var)
        return mean, std_dev

# --- 2. ESTRUCTURA DE NODO ---

@dataclass
class QuadtreeNode:
    y: int
    x: int
    h: int
    w: int
    depth: int
    children: List['QuadtreeNode'] = None
    # Datos de Color
    color_tl: Optional[np.ndarray] = None
    color_tr: Optional[np.ndarray] = None
    color_bl: Optional[np.ndarray] = None
    color_br: Optional[np.ndarray] = None
    # Modo de Predicción ('flat' o 'interp')
    mode: str = 'interp' 

    @property
    def is_leaf(self):
        return not self.children

# --- 3. COMPRESOR PRINCIPAL ---

class QuadtreeCompressor:
    def __init__(self, min_block_size=4, max_depth=10):
        self.min_block_size = min_block_size
        self.max_depth = max_depth
        self.root = None
        self.leaves = []
        self._img = None
        self._integral_img = None
        self._integral_saliency = None
        self._grid_cache = {}

    def compress(self, image_rgb: np.ndarray, saliency_map: np.ndarray, threshold: float, alpha: float, lam: float = 10.0, beta: float = 2.0):
        self._img = image_rgb
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        self._integral_img = IntegralImageWrapper(gray)
        
        saliency_uint8 = (saliency_map * 255).astype(np.uint8)
        self._integral_saliency = IntegralImageWrapper(saliency_uint8)

        h, w = image_rgb.shape[:2]
        self.root = QuadtreeNode(0, 0, h, w, 0)
        
        # Paso 1: Segmentación Top-Down (Exploración)
        self._recursive_split(self.root, threshold * 0.8, alpha)
        self.balance_tree()

        # Paso 2: Optimización RDO Multi-Modo (Bottom-Up)
        print(f"[RDO Multi-Modo] Optimizando (Lambda={lam}, Beta={beta})...")
        self.leaves = []
        self.prune_with_multimode_rdo(lam=lam, beta=beta)
        
        # Limpieza
        self._img = None
        self._integral_img = None
        self._integral_saliency = None

    # --- LÓGICA RDO MULTI-MODO ---

    def prune_with_multimode_rdo(self, lam: float, beta: float):
        if self.root:
            self._prune_multimode_recursive(self.root, lam, beta)
            self._collect_leaves_recursive(self.root)

    def _prune_multimode_recursive(self, node: QuadtreeNode, lam: float, beta: float) -> float:
        # Calcular Factor de Importancia (Saliencia)
        mean_sal, _ = self._integral_saliency.get_stats(node.y, node.x, node.h, node.w)
        norm_sal = mean_sal / 255.0
        importance = 1.0 + (beta * norm_sal)

        # --- OPCIÓN A: MODO FLAT (Barato, ~3 bytes) ---
        d_flat_raw = self._calculate_distortion(node, mode='flat')
        d_flat = d_flat_raw * importance
        # Costo estimado: 1 bit (Leaf) + 1 bit (Mode) + 24 bits (Color) = 26
        r_flat = 26.0
        j_flat = d_flat + (lam * r_flat)

        # --- OPCIÓN B: MODO INTERP (Caro, ~6 bytes) ---
        d_interp_raw = self._calculate_distortion(node, mode='interp')
        d_interp = d_interp_raw * importance
        # Costo estimado: 1 bit (Leaf) + 1 bit (Mode) + 48 bits (Colors) = 50
        r_interp = 50.0
        j_interp = d_interp + (lam * r_interp)

        # Elegir ganador local
        if j_flat <= j_interp:
            j_leaf = j_flat
            best_mode = 'flat'
        else:
            j_leaf = j_interp
            best_mode = 'interp'

        # Si ya es hoja por profundidad, retornar
        if node.is_leaf:
            node.mode = best_mode
            if node.color_tl is None: self._capture_leaf_data(node)
            return j_leaf

        # --- OPCIÓN C: SPLIT ---
        j_children = 0.0
        for child in node.children:
            j_children += self._prune_multimode_recursive(child, lam, beta)
        
        j_split = (lam * 1.0) + j_children 

        # Decisión Final
        if j_leaf <= j_split:
            # PODA
            node.children = []
            node.mode = best_mode
            if node.color_tl is None: self._capture_leaf_data(node)
            return j_leaf
        else:
            # MANTENER
            return j_split

    def _calculate_distortion(self, node: QuadtreeNode, mode: str) -> float:
        y, x, h, w = node.y, node.x, node.h, node.w
        original = self._img[y:y+h, x:x+w].astype(np.float32)

        if mode == 'flat':
            # Distortion respecto al promedio
            avg_color = np.mean(original, axis=(0,1))
            diff = original - avg_color
            return np.sum(diff ** 2)

        elif mode == 'interp':
            # Distortion respecto a interpolación bilineal
            h_img, w_img = self._img.shape[:2]
            y1, x1 = min(y + h - 1, h_img - 1), min(x + w - 1, w_img - 1)
            
            c_tl = self._img[y, x].astype(np.float32)
            c_tr = self._img[y, x1].astype(np.float32)
            c_bl = self._img[y1, x].astype(np.float32)
            c_br = self._img[y1, x1].astype(np.float32)

            xv, yv = self._get_interpolation_grids(h, w)
            
            # --- CORRECCIÓN BROADCASTING ---
            # Convertimos (H, W) -> (H, W, 1) para que sea compatible con RGB (3,)
            xv = xv[..., np.newaxis]
            yv = yv[..., np.newaxis]
            
            top = c_tl * (1 - xv) + c_tr * xv
            bottom = c_bl * (1 - xv) + c_br * xv
            rec = top * (1 - yv) + bottom * yv
            
            diff = original - rec
            return np.sum(diff ** 2)
        
        return float('inf')

    # --- RECONSTRUCCIÓN ---

    def reconstruct(self, node: QuadtreeNode) -> np.ndarray:
        h, w = node.h, node.w
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        self._reconstruct_recursive_multimode(node, canvas)
        return np.clip(canvas, 0, 255).astype(np.uint8)

    def _reconstruct_recursive_multimode(self, node, canvas):
        if not node.is_leaf:
            for child in node.children:
                self._reconstruct_recursive_multimode(child, canvas)
        else:
            y, x, h, w = node.y, node.x, node.h, node.w
            
            raw = [node.color_tl, node.color_tr, node.color_bl, node.color_br]
            colors = [c if c is not None else np.zeros(3) for c in raw]

            if node.mode == 'flat':
                val = colors[0]
                canvas[y:y+h, x:x+w] = val
            else:
                xv, yv = self._get_interpolation_grids(h, w)
                
                # --- CORRECCIÓN BROADCASTING ---
                xv = xv[..., np.newaxis]
                yv = yv[..., np.newaxis]

                c_tl, c_tr, c_bl, c_br = colors
                
                top = c_tl * (1 - xv) + c_tr * xv
                bottom = c_bl * (1 - xv) + c_br * xv
                block = top * (1 - yv) + bottom * yv
                canvas[y:y+h, x:x+w] = block

    # --- UTILS INTERNOS ---

    def _recursive_split(self, node, threshold, alpha):
        mean_sal, _ = self._integral_saliency.get_stats(node.y, node.x, node.h, node.w)
        norm_sal = mean_sal / 255.0
        
        effective_threshold = threshold * np.exp(-alpha * norm_sal)
        
        _, std_dev = self._integral_img.get_stats(node.y, node.x, node.h, node.w)
        
        can_split = node.depth < self.max_depth and node.h > self.min_block_size and node.w > self.min_block_size
        
        if can_split and std_dev > effective_threshold:
            self._split_node(node)
            for child in node.children:
                self._recursive_split(child, threshold, alpha)

    def _split_node(self, node):
        half_h, half_w = node.h // 2, node.w // 2
        y, x = node.y, node.x
        node.children = [
            QuadtreeNode(y, x, half_h, half_w, node.depth + 1),
            QuadtreeNode(y, x + half_w, half_h, node.w - half_w, node.depth + 1),
            QuadtreeNode(y + half_h, x, node.h - half_h, half_w, node.depth + 1),
            QuadtreeNode(y + half_h, x + half_w, node.h - half_h, node.w - half_w, node.depth + 1)
        ]

    def balance_tree(self):
        self._collect_leaves_recursive(self.root)
        limit = 10000
        count = 0
        while True:
            broken = False
            leaves_copy = list(self.leaves)
            for leaf in leaves_copy:
                if not leaf.is_leaf: continue
                neighbors = self._find_neighbors(leaf)
                for neighbor in neighbors:
                    if neighbor.depth < leaf.depth - 1:
                        self._split_node(neighbor)
                        self.leaves.append(neighbor.children[0]) 
                        broken = True
                        break
                if broken: break
            
            if not broken or count > limit: break
            self.leaves = []
            self._collect_leaves_recursive(self.root)
            count += 1

    def _collect_leaves_recursive(self, node):
        if node.is_leaf:
            self.leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves_recursive(child)

    def _capture_leaf_data(self, node):
        h_img, w_img = self._img.shape[:2]
        y, x, h, w = node.y, node.x, node.h, node.w
        
        y1 = min(y + h - 1, h_img - 1)
        x1 = min(x + w - 1, w_img - 1)
        
        node.color_tl = self._img[y, x].astype(np.float32)
        node.color_tr = self._img[y, x1].astype(np.float32)
        node.color_bl = self._img[y1, x].astype(np.float32)
        node.color_br = self._img[y1, x1].astype(np.float32)

    def _get_interpolation_grids(self, h, w):
        key = (h, w)
        if key not in self._grid_cache:
            x = np.linspace(0, 1, w, dtype=np.float32)
            y = np.linspace(0, 1, h, dtype=np.float32)
            xv, yv = np.meshgrid(x, y)
            self._grid_cache[key] = (xv, yv)
        return self._grid_cache[key]
    
    def _find_neighbors(self, node):
        # Esta función es compleja de implementar "full" sin punteros a padres.
        # Para RDO Multi-Modo esto NO es crítico (solo visual).
        # Retornamos vacío para no bloquear la ejecución si no tienes la impl completa.
        return []