import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional
import src.dct_utils as dct_utils

class IntegralImageWrapper:
    def __init__(self, img: np.ndarray):
        self.img = img.astype(np.float32)
        self._sat = cv2.integral(self.img)
        self._sat_sq = cv2.integral(self.img ** 2)

    def get_stats(self, y: int, x: int, h: int, w: int):
        y0, x0 = y, x
        y1, x1 = y + h, x + w
        area = h * w
        if area <= 0: return 0.0, 0.0
        s = (self._sat[y1, x1] - self._sat[y1, x0] - 
             self._sat[y0, x1] + self._sat[y0, x0])
        sq = (self._sat_sq[y1, x1] - self._sat_sq[y1, x0] - 
              self._sat_sq[y0, x1] + self._sat_sq[y0, x0])
        mean = s / area
        var = (sq / area) - (mean ** 2)
        if var < 0: var = 0
        return mean, np.sqrt(var)

@dataclass
class QuadtreeNode:
    y: int
    x: int
    h: int
    w: int
    depth: int
    children: List['QuadtreeNode'] = None
    color_tl: Optional[np.ndarray] = None
    color_tr: Optional[np.ndarray] = None
    color_bl: Optional[np.ndarray] = None
    color_br: Optional[np.ndarray] = None
    dct_coeffs: Optional[np.ndarray] = None
    mode: str = 'interp' 
    @property
    def is_leaf(self): return not self.children

class QuadtreeCompressor:
    def __init__(self, min_block_size=8, max_depth=10):
        self.min_block_size = max(8, min_block_size)
        self.max_depth = max_depth
        self.root = None
        self.leaves = []
        self._img = None
        self._integral_img = None
        self._integral_saliency = None
        self._grid_cache = {}
        self.orig_h = 0
        self.orig_w = 0
        # Guardaremos la calidad calculada aquí
        self.dynamic_quality = 50 

    def compress(self, image_rgb: np.ndarray, saliency_map: np.ndarray, threshold: float, alpha: float, lam: float = 10.0, beta: float = 2.0):
        self.orig_h, self.orig_w = image_rgb.shape[:2]
        
        # --- MEJORA 6: CÁLCULO DINÁMICO DE CALIDAD ---
        # Fórmula Logarítmica: A mayor Lambda, menor calidad.
        # +1 evita log(0). 
        # El factor 12.0 es empírico para mapear el rango [1, 10000] a [95, 5].
        raw_q = 100 - 12.0 * math.log(lam + 1)
        self.dynamic_quality = int(max(5, min(95, raw_q)))
        
        # Padding
        max_dim = max(self.orig_h, self.orig_w)
        next_pow2 = 2 ** math.ceil(math.log2(max_dim))
        next_pow2 = max(next_pow2, 16)
        
        self._img = self._pad_image(image_rgb, next_pow2, next_pow2)
        saliency_padded = self._pad_image(saliency_map, next_pow2, next_pow2)
        
        gray = cv2.cvtColor(self._img, cv2.COLOR_RGB2GRAY)
        self._integral_img = IntegralImageWrapper(gray)
        saliency_uint8 = (saliency_padded * 255).astype(np.uint8)
        self._integral_saliency = IntegralImageWrapper(saliency_uint8)

        h, w = self._img.shape[:2]
        self.root = QuadtreeNode(0, 0, h, w, 0)
        
        self._recursive_split(self.root, threshold * 0.8, alpha)
        
        # Debug para verificar que la mejora funciona
        # print(f"[RDO] Lambda={lam} -> Quality Factor={self.dynamic_quality}")
        
        self.leaves = []
        self.prune_with_multimode_rdo(lam=lam, beta=beta)
        
        self._img = None
        self._integral_img = None
        self._integral_saliency = None

    def _pad_image(self, img, target_h, target_w):
        h, w = img.shape[:2]
        if h == target_h and w == target_w: return img
        if len(img.shape) == 3:
            padded = np.zeros((target_h, target_w, 3), dtype=img.dtype)
            padded[:h, :w, :] = img
        else:
            padded = np.zeros((target_h, target_w), dtype=img.dtype)
            padded[:h, :w] = img
        return padded

    def prune_with_multimode_rdo(self, lam: float, beta: float):
        if self.root:
            self._prune_multimode_recursive(self.root, lam, beta)
            self._collect_leaves_recursive(self.root)

    def _prune_multimode_recursive(self, node: QuadtreeNode, lam: float, beta: float) -> float:
        if node.x >= self.orig_w or node.y >= self.orig_h:
            node.children = []
            node.mode = 'flat'
            node.color_tl = np.zeros(3, dtype=np.float32)
            return 0.1
        
        mean_sal, _ = self._integral_saliency.get_stats(node.y, node.x, node.h, node.w)
        norm_sal = mean_sal / 255.0
        importance = 1.0 + (beta * norm_sal)

        d_flat = self._calculate_distortion(node, 'flat') * importance
        j_flat = d_flat + (lam * 26.0)

        d_interp = self._calculate_distortion(node, 'interp') * importance
        j_interp = d_interp + (lam * 50.0)

        j_dct = float('inf')
        coeffs_candidate = None

        if node.h == 8 and node.w == 8:
            y, x, h, w = node.y, node.x, node.h, node.w
            block_rgb = self._img[y:y+h, x:x+w]
            block_gray = cv2.cvtColor(block_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            block_centered = block_gray - 128.0

            # USAMOS LA CALIDAD DINÁMICA AQUÍ
            q_matrix = dct_utils.get_quantization_matrix(self.dynamic_quality)
            
            coeffs_candidate = dct_utils.dct_transform_block(block_centered, q_matrix)
            rec_gray = dct_utils.idct_reconstruct_block(coeffs_candidate, q_matrix) + 128.0
            
            diff_dct = block_gray - rec_gray
            d_dct = np.sum(diff_dct ** 2) * importance
            
            non_zeros = np.count_nonzero(coeffs_candidate)
            r_dct = 8.0 + (non_zeros * 3.0) 
            j_dct = d_dct + (lam * r_dct)

        best_j = min(j_flat, j_interp, j_dct)

        if best_j == j_dct: best_mode = 'dct'
        elif best_j == j_flat: best_mode = 'flat'
        else: best_mode = 'interp'

        if node.is_leaf:
            self._assign_mode_data(node, best_mode, coeffs_candidate)
            return best_j

        j_children = 0.0
        for child in node.children:
            j_children += self._prune_multimode_recursive(child, lam, beta)
        
        j_split = (lam * 2.0) + j_children 

        if best_j <= j_split:
            node.children = []
            self._assign_mode_data(node, best_mode, coeffs_candidate)
            return best_j
        else:
            return j_split
        
    def _assign_mode_data(self, node, mode, coeffs):
        node.mode = mode
        if mode == 'dct': node.dct_coeffs = coeffs
        self._capture_leaf_color(node)

    def _calculate_distortion(self, node, mode):
        y, x, h, w = node.y, node.x, node.h, node.w
        original = self._img[y:y+h, x:x+w].astype(np.float32)

        if mode == 'flat':
            avg = np.mean(original, axis=(0,1))
            return np.sum((original - avg) ** 2)
        elif mode == 'interp':
            xv, yv = self._get_interpolation_grids(h, w)
            xv, yv = xv[..., None], yv[..., None]
            c_tl = self._img[y, x].astype(float)
            c_tr = self._img[y, x+w-1].astype(float)
            c_bl = self._img[y+h-1, x].astype(float)
            c_br = self._img[y+h-1, x+w-1].astype(float)
            top = c_tl*(1-xv) + c_tr*xv
            bot = c_bl*(1-xv) + c_br*xv
            rec = top*(1-yv) + bot*yv
            return np.sum((original - rec) ** 2)
        return float('inf')

    # --- RECONSTRUCCIÓN CON CALIDAD VARIABLE ---
    def reconstruct(self, node: QuadtreeNode, override_quality: int = None) -> np.ndarray:
        # Si venimos de decodificación, usamos el valor leído del archivo
        # Si estamos en memoria, usamos self.dynamic_quality
        q_factor = override_quality if override_quality is not None else self.dynamic_quality
        
        h_pad, w_pad = node.h, node.w
        canvas = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
        self._reconstruct_recursive_multimode(node, canvas, q_factor)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        
        if self.orig_h > 0 and self.orig_w > 0:
            return canvas[:self.orig_h, :self.orig_w]
        return canvas

    def _reconstruct_recursive_multimode(self, node, canvas, q_factor):
        if not node.is_leaf:
            for child in node.children:
                self._reconstruct_recursive_multimode(child, canvas, q_factor)
        else:
            y, x, h, w = node.y, node.x, node.h, node.w
            
            if node.mode == 'dct' and node.dct_coeffs is not None:
                # USAMOS EL Q FACTOR PASADO COMO ARGUMENTO
                q_matrix = dct_utils.get_quantization_matrix(q_factor)
                
                rec_y = dct_utils.idct_reconstruct_block(node.dct_coeffs, q_matrix, shape=(h,w)) + 128.0
                rec_y = np.clip(rec_y, 0, 255).astype(np.float32)

                if node.color_tl is not None:
                    flat_rgb = node.color_tl.astype(np.uint8).reshape(1,1,3)
                    flat_ycrcb = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2YCrCb)
                    cb_val = float(flat_ycrcb[0,0,1])
                    cr_val = float(flat_ycrcb[0,0,2])
                else:
                    cb_val, cr_val = 128.0, 128.0

                curr_h, curr_w = rec_y.shape[:2]
                cb_block = np.full((curr_h, curr_w), cb_val, dtype=np.float32)
                cr_block = np.full((curr_h, curr_w), cr_val, dtype=np.float32)
                
                merged = cv2.merge([rec_y, cb_block, cr_block])
                rgb_block = cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
                
                if rgb_block.shape[:2] != (h, w):
                    rgb_block = cv2.resize(rgb_block, (w, h))
                canvas[y:y+h, x:x+w] = rgb_block.astype(np.float32)

            elif node.mode == 'flat':
                val = node.color_tl if node.color_tl is not None else np.zeros(3)
                canvas[y:y+h, x:x+w] = val

            else: # Interp
                xv, yv = self._get_interpolation_grids(h, w)
                xv, yv = xv[..., None], yv[..., None]
                c_tl = node.color_tl if node.color_tl is not None else np.zeros(3)
                c_tr = node.color_tr if node.color_tr is not None else c_tl
                c_bl = node.color_bl if node.color_bl is not None else c_tl
                c_br = node.color_br if node.color_br is not None else c_tl
                top = c_tl*(1-xv) + c_tr*xv
                bot = c_bl*(1-xv) + c_br*xv
                block = top*(1-yv) + bot*yv
                canvas[y:y+h, x:x+w] = block

    def _recursive_split(self, node, threshold, alpha):
        mean_sal, _ = self._integral_saliency.get_stats(node.y, node.x, node.h, node.w)
        eff_th = threshold * np.exp(-alpha * (mean_sal / 255.0))
        _, std = self._integral_img.get_stats(node.y, node.x, node.h, node.w)
        can_split = node.depth < self.max_depth and node.h > self.min_block_size
        if can_split and std > eff_th:
            self._split_node(node)
            for child in node.children: self._recursive_split(child, threshold, alpha)

    def _split_node(self, node):
        h2, w2 = node.h // 2, node.w // 2
        y, x = node.y, node.x
        d = node.depth + 1
        node.children = [
            QuadtreeNode(y, x, h2, w2, d), QuadtreeNode(y, x+w2, h2, w2, d),
            QuadtreeNode(y+h2, x, h2, w2, d), QuadtreeNode(y+h2, x+w2, h2, w2, d)
        ]

    def _collect_leaves_recursive(self, node):
        if node.is_leaf: self.leaves.append(node)
        else:
            for c in node.children: self._collect_leaves_recursive(c)

    def _capture_leaf_color(self, node):
        y, x, h, w = node.y, node.x, node.h, node.w
        c_tl = self._img[y, x]
        c_tr = self._img[y, x+w-1]
        c_bl = self._img[y+h-1, x]
        c_br = self._img[y+h-1, x+w-1]
        if node.mode in ['flat', 'dct']:
            avg = np.mean(self._img[y:y+h, x:x+w], axis=(0,1))
            node.color_tl = avg.astype(np.float32)
        else:
            node.color_tl = c_tl.astype(np.float32)
            node.color_tr = c_tr.astype(np.float32)
            node.color_bl = c_bl.astype(np.float32)
            node.color_br = c_br.astype(np.float32)

    def _get_interpolation_grids(self, h, w):
        key = (h, w)
        if key not in self._grid_cache:
            x = np.linspace(0, 1, w, dtype=np.float32)
            y = np.linspace(0, 1, h, dtype=np.float32)
            xv, yv = np.meshgrid(x, y)
            self._grid_cache[key] = (xv, yv)
        return self._grid_cache[key]