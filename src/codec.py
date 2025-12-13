import numpy as np
import zlib
import struct
from src.quadtree import QuadtreeNode

class QuadtreeCodec:
    """
    Codificador de Entropía para Quadtrees con soporte para múltiples esquemas de color.
    
    Modos soportados:
    1. 'flat': Guarda 1 color promedio por hoja (3 bytes). Uso: Standard QT.
    2. 'gradient': Guarda 4 colores RGB por hoja (12 bytes). Uso: Tu método (Original).
    3. 'optimized': Guarda 4 Luma + 1 Chroma promedio (6 bytes). Uso: Tu método (Optimizado).
       - Aplica subsampling 4:2:0 (estilo JPEG).
       - Reduce el tamaño del archivo al 50% comparado con 'gradient' manteniendo estructura.
    """

    def compress(self, root: QuadtreeNode, shape: tuple, mode: str = 'gradient') -> bytes:
        """
        Serializa y comprime el árbol.
        Args:
            root: Nodo raíz.
            shape: (Alto, Ancho).
            mode: 'flat', 'gradient', o 'optimized'.
        Returns:
            bytes: Stream binario comprimido con zlib.
        """
        h, w = shape
        # Header: Alto (2 bytes), Ancho (2 bytes), Modo (1 byte)
        # Codificamos el modo como entero: 0=flat, 1=gradient, 2=optimized
        mode_map = {'flat': 0, 'gradient': 1, 'optimized': 2}
        mode_byte = mode_map.get(mode, 1)
        
        header = struct.pack('>HHB', h, w, mode_byte)
        
        stream = bytearray()
        self._encode_node(root, stream, mode)
        
        full_data = header + stream
        # Nivel 9 para máxima compresión DEFLATE
        compressed_data = zlib.compress(full_data, level=9)
        
        return compressed_data

    def decompress(self, compressed_data: bytes):
        """Reconstruye el árbol desde bytes comprimidos."""
        try:
            full_data = zlib.decompress(compressed_data)
        except zlib.error:
            raise ValueError("Datos corruptos o formato zlib inválido.")

        # Leer Header (5 bytes)
        h, w, mode_byte = struct.unpack('>HHB', full_data[:5])
        
        # Mapeo inverso de modo
        modes = {0: 'flat', 1: 'gradient', 2: 'optimized'}
        mode = modes.get(mode_byte, 'gradient')
        
        # Reconstruir Árbol
        stream_iter = iter(full_data[5:])
        root = self._decode_node(stream_iter, 0, 0, h, w, 0, mode)
        
        return root, (h, w)

    def _encode_node(self, node: QuadtreeNode, stream: bytearray, mode: str):
        """Recorrido Pre-Order para serialización."""
        if not node.is_leaf:
            stream.append(0) # Flag: Split
            for child in node.children:
                self._encode_node(child, stream, mode)
        else:
            stream.append(1) # Flag: Leaf
            
            # Obtener los 4 colores (o rellenar con negro si es None)
            raw_colors = [node.color_tl, node.color_tr, node.color_bl, node.color_br]
            colors = [c if c is not None else np.zeros(3, dtype=np.float32) for c in raw_colors]

            if mode == 'flat':
                # MODO BLOQUE (3 Bytes)
                # Promedio de las esquinas -> 1 color RGB
                avg = np.mean(colors, axis=0)
                stream.extend(np.clip(avg, 0, 255).astype(np.uint8).tobytes())

            elif mode == 'gradient':
                # MODO FULL RGB (12 Bytes)
                # 4 esquinas * 3 canales
                for col in colors:
                    stream.extend(np.clip(col, 0, 255).astype(np.uint8).tobytes())

            elif mode == 'optimized':
                # MODO YCbCr 4:2:0 (6 Bytes) - INGENIERÍA PURA
                # Teoría: El ojo es sensible al brillo (Y) pero poco al color (CbCr).
                # Guardamos 4 Y (detalle espacial) y solo 1 par CbCr (promedio de color).
                
                ys = []
                cbs = []
                crs = []
                
                for col in colors:
                    r, g, b = col[0], col[1], col[2]
                    # Fórmulas estándar ITU-R BT.601
                    y  =  0.299*r + 0.587*g + 0.114*b
                    cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
                    cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
                    
                    ys.append(y)
                    cbs.append(cb)
                    crs.append(cr)
                
                # 1. Escribimos los 4 valores de Luma (Brillo) - Detalle puro
                stream.extend(np.clip(ys, 0, 255).astype(np.uint8).tobytes())
                
                # 2. Escribimos el promedio de Chroma (Color)
                avg_cb = sum(cbs) / 4.0
                avg_cr = sum(crs) / 4.0
                stream.append(int(np.clip(avg_cb, 0, 255)))
                stream.append(int(np.clip(avg_cr, 0, 255)))

    def _decode_node(self, stream, y, x, h, w, depth, mode):
        """Reconstrucción recursiva."""
        try:
            flag = next(stream)
        except StopIteration:
            return None

        node = QuadtreeNode(y, x, h, w, depth)

        if flag == 0: # Split
            half_h, half_w = h // 2, w // 2
            node.children = [
                self._decode_node(stream, y, x, half_h, half_w, depth+1, mode),           # TL
                self._decode_node(stream, y, x+half_w, half_h, w-half_w, depth+1, mode),  # TR
                self._decode_node(stream, y+half_h, x, h-half_h, half_w, depth+1, mode),  # BL
                self._decode_node(stream, y+half_h, x+half_w, h-half_h, w-half_w, depth+1, mode) # BR
            ]
        
        elif flag == 1: # Leaf
            if mode == 'flat':
                # Leemos 3 bytes, asignamos el mismo color a las 4 esquinas
                r, g, b = next(stream), next(stream), next(stream)
                col = np.array([r, g, b], dtype=np.float32)
                node.color_tl = node.color_tr = node.color_bl = node.color_br = col
            
            elif mode == 'gradient':
                # Leemos 12 bytes
                node.color_tl = self._read_rgb(stream)
                node.color_tr = self._read_rgb(stream)
                node.color_bl = self._read_rgb(stream)
                node.color_br = self._read_rgb(stream)
            
            elif mode == 'optimized':
                # Leemos 6 bytes: 4 Ys + 1 Cb + 1 Cr
                y_tl = next(stream)
                y_tr = next(stream)
                y_bl = next(stream)
                y_br = next(stream)
                cb = next(stream)
                cr = next(stream)
                
                # Reconstruimos RGB para cada esquina usando su Y propio y el CbCr compartido
                node.color_tl = self._ycbcr_to_rgb(y_tl, cb, cr)
                node.color_tr = self._ycbcr_to_rgb(y_tr, cb, cr)
                node.color_bl = self._ycbcr_to_rgb(y_bl, cb, cr)
                node.color_br = self._ycbcr_to_rgb(y_br, cb, cr)

        return node

    def _read_rgb(self, stream):
        return np.array([next(stream), next(stream), next(stream)], dtype=np.float32)

    def _ycbcr_to_rgb(self, y, cb, cr):
        """Conversión inversa YCbCr -> RGB."""
        # Fórmulas estándar
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return np.array([r, g, b], dtype=np.float32)