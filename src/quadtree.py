import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional
import src.dct_utils as dct_utils

class IntegralImageWrapper:
    """
    Optimized wrapper for O(1) sum and O(1) SSE calculation.
    Formula: Sum((x - mean)^2) = Sum(x^2) - (Sum(x)^2)/N
    """
    def __init__(self, img: np.ndarray):
        self.img = img.astype(np.float32)
        # OpenCV integral añade una fila y columna extra de ceros al inicio
        self._sat = cv2.integral(self.img)
        self._sat_sq = cv2.integral(self.img ** 2)

    def get_stats(self, y: int, x: int, h: int, w: int):
        y0, x0 = y, x
        y1, x1 = y + h, x + w
        area = h * w
        if area <= 0: return 0.0, 0.0
        
        # Access optimization
        s = (self._sat[y1, x1] - self._sat[y1, x0] - self._sat[y0, x1] + self._sat[y0, x0])
        
        # Retornamos solo la media y 0.0 (std) por compatibilidad si no se requiere varianza aquí
        return s / area, 0.0 

    def get_sum_and_sq_sum(self, y: int, x: int, h: int, w: int):
        y0, x0 = y, x
        y1, x1 = y + h, x + w
        
        # Direct array access
        s = (self._sat[y1, x1] - self._sat[y1, x0] - self._sat[y0, x1] + self._sat[y0, x0])
        sq = (self._sat_sq[y1, x1] - self._sat_sq[y1, x0] - self._sat_sq[y0, x1] + self._sat_sq[y0, x0])
        return s, sq

    def get_sse_flat(self, y, x, h, w):
        """Returns Sum Squared Error relative to the mean of the block in O(1)."""
        # --- CORRECCIÓN AQUÍ ---
        y0, x0 = y, x
        y1, x1 = y + h, x + w
        # -----------------------

        area = h * w
        if area <= 0: return 0.0
        
        s = (self._sat[y1, x1] - self._sat[y1, x0] - self._sat[y0, x1] + self._sat[y0, x0])
        sq = (self._sat_sq[y1, x1] - self._sat_sq[y1, x0] - self._sat_sq[y0, x1] + self._sat_sq[y0, x0])
        
        # Formula: SSE = Sum(x^2) - (Sum(x)^2) / N
        return abs(sq - (s*s)/area)

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
        self.dynamic_quality = 50 
        
        # Cache for batch DCT results
        self._dct_cache_map = {} 

    def compress(self, image_rgb: np.ndarray, saliency_map: np.ndarray, threshold: float, alpha: float, lam: float = 10.0, beta: float = 2.0):
        self.orig_h, self.orig_w = image_rgb.shape[:2]
        
        # Dynamic Quality Calculation
        raw_q = 100 - 12.0 * math.log(lam + 1)
        self.dynamic_quality = int(max(5, min(95, raw_q)))
        
        # Padding
        max_dim = max(self.orig_h, self.orig_w)
        # Fast next power of 2
        next_pow2 = 1 << (max_dim - 1).bit_length() 
        next_pow2 = max(next_pow2, 16)
        
        self._img = self._pad_image(image_rgb, next_pow2, next_pow2)
        saliency_padded = self._pad_image(saliency_map, next_pow2, next_pow2)
        
        gray = cv2.cvtColor(self._img, cv2.COLOR_RGB2GRAY)
        self._integral_img = IntegralImageWrapper(gray)
        
        # Saliency Integral
        self._integral_saliency = IntegralImageWrapper((saliency_padded * 255).astype(np.float32))

        # --- PRE-CALCULATE DCT GRID (VECTORIZED) ---
        self._precompute_dct_grid(gray, self.dynamic_quality)

        h, w = self._img.shape[:2]
        self.root = QuadtreeNode(0, 0, h, w, 0)
        
        # Recursive Split
        self._recursive_split(self.root, threshold * 0.8, alpha)
        
        self.leaves = []
        # Pruning (RDO)
        self.prune_with_multimode_rdo(lam=lam, beta=beta)
        
        # Cleanup large arrays
        self._img = None
        self._integral_img = None
        self._integral_saliency = None
        self._dct_cache_map = {}

    def _precompute_dct_grid(self, gray_img, quality):
        """
        Slices the image into 8x8 blocks, runs Batch DCT, and stores results for fast lookup.
        """
        h, w = gray_img.shape
        h_blocks = h // 8
        w_blocks = w // 8
        
        # View as blocks: (N_blocks, 8, 8)
        # Transpose es necesario porque reshape "consume" los datos fila por fila.
        # Queremos bloques (y, x) -> (block_y, block_x, 8, 8)
        blocks = gray_img.reshape(h_blocks, 8, w_blocks, 8).transpose(0, 2, 1, 3)
        flat_blocks_input = blocks.reshape(-1, 8, 8) - 128.0 # Centered
        
        q_matrix = dct_utils.get_quantization_matrix(quality)
        
        # --- VECTORIZED CALL ---
        coeffs, sse_errors, non_zeros = dct_utils.batch_dct_transform(flat_blocks_input, q_matrix)
        
        # Store in flat arrays, access via index
        self._dct_flat_coeffs = coeffs
        self._dct_sse_errors = sse_errors
        self._dct_non_zeros = non_zeros
        self._dct_grid_w = w_blocks

    def _get_dct_data_fast(self, y, x):
        """O(1) lookup for DCT precalculated data"""
        bx = x // 8
        by = y // 8
        idx = by * self._dct_grid_w + bx
        # Safety check
        if idx < 0 or idx >= len(self._dct_sse_errors):
            return float('inf'), None, 0
        return self._dct_sse_errors[idx], self._dct_flat_coeffs[idx], self._dct_non_zeros[idx]

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

    def _collect_leaves_recursive(self, node):
        if node.is_leaf: self.leaves.append(node)
        else:
            for c in node.children: self._collect_leaves_recursive(c)

    def _prune_multimode_recursive(self, node: QuadtreeNode, lam: float, beta: float) -> float:
        if node.x >= self.orig_w or node.y >= self.orig_h:
            node.children = []
            node.mode = 'flat'
            return 0.1 # Minimal cost for out-of-bounds
        
        # 1. Saliency via Integral Image O(1)
        s_sum, _ = self._integral_saliency.get_sum_and_sq_sum(node.y, node.x, node.h, node.w)
        mean_sal = s_sum / (node.h * node.w)
        importance = 1.0 + (beta * (mean_sal / 255.0))

        # 2. Flat Cost: Integral Image O(1)
        d_flat = self._integral_img.get_sse_flat(node.y, node.x, node.h, node.w) * importance
        j_flat = d_flat + (lam * 26.0)

        # 3. Interp Cost: Vectorized calculation
        d_interp = self._calculate_distortion_interp(node) * importance
        j_interp = d_interp + (lam * 50.0)

        # 4. DCT Cost: O(1) Lookup
        j_dct = float('inf')
        coeffs_candidate = None

        if node.h == 8 and node.w == 8:
            d_dct_raw, coeffs_candidate, non_zeros = self._get_dct_data_fast(node.y, node.x)
            d_dct = d_dct_raw * importance
            r_dct = 8.0 + (non_zeros * 3.0) 
            j_dct = d_dct + (lam * r_dct)

        best_j = min(j_flat, j_interp, j_dct)

        if best_j == j_dct: best_mode = 'dct'
        elif best_j == j_flat: best_mode = 'flat'
        else: best_mode = 'interp'

        if node.is_leaf:
            self._assign_mode_data(node, best_mode, coeffs_candidate)
            return best_j

        # Recursive step
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

    def _calculate_distortion_interp(self, node):
        """Optimized interpolation distortion using NumPy broadcasting."""
        y, x, h, w = node.y, node.x, node.h, node.w
        
        # Extract block once
        original = self._img[y:y+h, x:x+w].astype(np.float32) 
        # We need grayscale for distortion metric usually
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if original.shape[2] == 3 else original

        xv, yv = self._get_interpolation_grids(h, w)
        
        # Corners (using gray values from integral image source for consistency)
        img_gray_full = self._integral_img.img 
        c_tl = img_gray_full[y, x]
        c_tr = img_gray_full[y, x+w-1]
        c_bl = img_gray_full[y+h-1, x]
        c_br = img_gray_full[y+h-1, x+w-1]

        # Vectorized bilinear
        top = c_tl*(1-xv) + c_tr*xv
        bot = c_bl*(1-xv) + c_br*xv
        rec = top*(1-yv) + bot*yv
        
        return np.sum((orig_gray - rec) ** 2)

    def reconstruct(self, node: QuadtreeNode, override_quality: int = None) -> np.ndarray:
        q_factor = override_quality if override_quality is not None else self.dynamic_quality
        h_pad, w_pad = node.h, node.w
        
        # Canvas as float for accumulation
        canvas_f = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
        
        self._reconstruct_recursive_multimode(node, canvas_f, q_factor)
        
        canvas = np.clip(canvas_f, 0, 255).astype(np.uint8)
        if self.orig_h > 0 and self.orig_w > 0:
            return canvas[:self.orig_h, :self.orig_w]
        return canvas

    def _reconstruct_recursive_multimode(self, node, canvas, q_factor):
        """
        Optimization: Instead of pixel-wise loops, use slice assignment (canvas[y:y+h] = block)
        """
        if not node.is_leaf:
            for child in node.children:
                self._reconstruct_recursive_multimode(child, canvas, q_factor)
            return

        y, x, h, w = node.y, node.x, node.h, node.w
        
        if node.mode == 'dct' and node.dct_coeffs is not None:
            q_matrix = dct_utils.get_quantization_matrix(q_factor)
            rec_y = dct_utils.idct_reconstruct_block(node.dct_coeffs, q_matrix, shape=(h,w)) + 128.0
            rec_y = np.clip(rec_y, 0, 255)

            xv, yv = self._get_interpolation_grids(h, w)
            
            def get_chroma(color):
                if color is None: return 128.0, 128.0
                r, g, b = color
                return (128 - 0.168736*r - 0.331264*g + 0.5*b, 
                        128 + 0.5*r - 0.418688*g - 0.081312*b)

            cb_tl, cr_tl = get_chroma(node.color_tl)
            cb_tr, cr_tr = get_chroma(node.color_tr)
            cb_bl, cr_bl = get_chroma(node.color_bl)
            cb_br, cr_br = get_chroma(node.color_br)
            
            # Interpolación bilineal para canales Cb y Cr
            top_cb = cb_tl*(1-xv) + cb_tr*xv
            bot_cb = cb_bl*(1-xv) + cb_br*xv
            cb_block = top_cb*(1-yv) + bot_cb*yv
            
            top_cr = cr_tl*(1-xv) + cr_tr*xv
            bot_cr = cr_bl*(1-xv) + cr_br*xv
            cr_block = top_cr*(1-yv) + bot_cr*yv

            # YCrCb -> RGB linear transformation vectorial
            c_term_r = 1.402 * (cr_block - 128)
            c_term_g = -0.344136 * (cb_block - 128) - 0.714136 * (cr_block - 128)
            c_term_b = 1.772 * (cb_block - 128)
            
            canvas[y:y+h, x:x+w, 0] = rec_y + c_term_r
            canvas[y:y+h, x:x+w, 1] = rec_y + c_term_g
            canvas[y:y+h, x:x+w, 2] = rec_y + c_term_b

        elif node.mode == 'flat':
            val = node.color_tl if node.color_tl is not None else np.zeros(3)
            canvas[y:y+h, x:x+w] = val 

        else: # Interp
            xv, yv = self._get_interpolation_grids(h, w)
            xv = xv[..., None] # Broadcast to 3 channels
            yv = yv[..., None]
            
            c_tl = node.color_tl if node.color_tl is not None else np.zeros(3)
            c_tr = node.color_tr if node.color_tr is not None else c_tl
            c_bl = node.color_bl if node.color_bl is not None else c_tl
            c_br = node.color_br if node.color_br is not None else c_tl
            
            top = c_tl*(1-xv) + c_tr*xv
            bot = c_bl*(1-xv) + c_br*xv
            block = top*(1-yv) + bot*yv
            canvas[y:y+h, x:x+w] = block

    def _recursive_split(self, node, threshold, alpha):
        s, sq = self._integral_saliency.get_sum_and_sq_sum(node.y, node.x, node.h, node.w)
        mean_sal = s / (node.h * node.w)
        
        eff_th = threshold * np.exp(-alpha * (mean_sal / 255.0))
        
        # Calculate std dev of Image from integral
        s_img, sq_img = self._integral_img.get_sum_and_sq_sum(node.y, node.x, node.h, node.w)
        area = node.h * node.w
        var = (sq_img / area) - (s_img / area)**2
        std = np.sqrt(max(0, var))
        
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

    def _capture_leaf_color(self, node):
        y, x, h, w = node.y, node.x, node.h, node.w
        if node.mode == 'flat': 
            avg = np.mean(self._img[y:y+h, x:x+w], axis=(0,1))
            node.color_tl = avg.astype(np.float32)
        else: # Ahora 'interp' y 'dct' guardan las 4 esquinas
            node.color_tl = self._img[y, x].astype(np.float32)
            node.color_tr = self._img[y, x+w-1].astype(np.float32)
            node.color_bl = self._img[y+h-1, x].astype(np.float32)
            node.color_br = self._img[y+h-1, x+w-1].astype(np.float32)

    def _get_interpolation_grids(self, h, w):
        key = (h, w)
        if key not in self._grid_cache:
            x = np.linspace(0, 1, w, dtype=np.float32)
            y = np.linspace(0, 1, h, dtype=np.float32)
            xv, yv = np.meshgrid(x, y)
            self._grid_cache[key] = (xv, yv)
        return self._grid_cache[key]