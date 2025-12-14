import numpy as np
import zlib
import struct
from src.quadtree import QuadtreeNode

class QuadtreeCodec:
    """
    Codificador de Entropía Avanzado para Quadtrees.
    
    Características SOTA (State-of-the-Art):
    1. Bit-Packed Structure: La topología del árbol se guarda como un stream de bits.
    2. Stream Separation: Separa topología y color para maximizar la eficiencia DEFLATE.
    3. DPCM (Differential Pulse Code Modulation): Codifica diferencias relativas en lugar de absolutas.
    4. Chroma Subsampling (4:2:0): En modo 'optimized', reduce la resolución de color manteniendo el brillo.
    """

    def compress(self, root: QuadtreeNode, shape: tuple, mode: str = 'optimized') -> bytes:
        """
        Comprime el árbol usando Bit Packing + DPCM + Zlib.
        """
        h, w = shape
        
        # 1. Inicializar Predictores DPCM (Estado inicial)
        # Se reinician para cada imagen.
        self.prev_dc = np.array([0.0, 0.0, 0.0], dtype=np.float32) # Para modos RGB
        self.prev_y  = 0.0   # Para Luma
        self.prev_cb = 128.0 # Para Chroma Blue (gris neutro)
        self.prev_cr = 128.0 # Para Chroma Red (gris neutro)

        # 2. Configuración de Header
        mode_map = {'flat': 0, 'gradient': 1, 'optimized': 2}
        mode_byte = mode_map.get(mode, 2)
        
        # 3. Recorrido del Árbol (Separando bits de estructura y bytes de color)
        structure_bits = []      # Lista temporal de 0s y 1s
        color_data = bytearray() # Buffer de bytes para los colores (DPCM aplicado)
        
        self._encode_recursive(root, structure_bits, color_data, mode)
        
        # 4. Bit Packing (Convertir lista de bits a bytes reales)
        packed_structure = self._pack_bits(structure_bits)
        
        # 5. Ensamblaje del Payload Binario
        # Header: [Alto (2B)] [Ancho (2B)] [Modo (1B)]
        header = struct.pack('>HHB', h, w, mode_byte)
        
        # Tamaño de la estructura (4 bytes) - Necesario para el decodificador
        struct_len = struct.pack('>I', len(packed_structure))
        
        # Payload: [Header] + [LenStruct] + [StructureBytes] + [ColorBytes]
        full_payload = header + struct_len + packed_structure + color_data
        
        # 6. Compresión Final DEFLATE (Nivel 9 - Máximo)
        # Aquí es donde DPCM brilla: zlib comprime muy bien las diferencias pequeñas.
        return zlib.compress(full_payload, level=9)

    def decompress(self, compressed_data: bytes):
        """Descomprime y reconstruye el Quadtree."""
        try:
            full_payload = zlib.decompress(compressed_data)
        except zlib.error:
            raise ValueError("Datos corruptos o formato zlib inválido.")

        # 1. Inicializar Predictores DPCM para decodificación
        self.prev_dc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_y  = 0.0
        self.prev_cb = 128.0
        self.prev_cr = 128.0

        ptr = 0
        
        # 2. Leer Header
        h, w, mode_byte = struct.unpack('>HHB', full_payload[ptr:ptr+5])
        ptr += 5
        
        modes = {0: 'flat', 1: 'gradient', 2: 'optimized'}
        mode = modes.get(mode_byte, 'optimized')
        
        # 3. Leer longitud de estructura
        struct_len = struct.unpack('>I', full_payload[ptr:ptr+4])[0]
        ptr += 4
        
        # 4. Separar Streams
        packed_structure = full_payload[ptr : ptr + struct_len]
        ptr += struct_len
        raw_colors = full_payload[ptr:] # El resto es el stream de colores DPCM
        
        # 5. Crear Iteradores
        bit_stream = self._bit_generator(packed_structure)
        color_stream = iter(raw_colors)
        
        # 6. Reconstrucción Recursiva
        root = self._decode_recursive(bit_stream, color_stream, 0, 0, h, w, 0, mode)
        
        return root, (h, w)

    # --- Lógica de Recorrido (Estructura) ---

    def _encode_recursive(self, node: QuadtreeNode, bits: list, colors: bytearray, mode: str):
        if not node.is_leaf:
            bits.append(0) # Flag: Split (1 bit)
            for child in node.children:
                self._encode_recursive(child, bits, colors, mode)
        else:
            bits.append(1) # Flag: Leaf (1 bit)
            self._write_colors_dpcm(node, colors, mode)

    def _decode_recursive(self, bit_stream, color_stream, y, x, h, w, depth, mode):
        try:
            flag = next(bit_stream)
        except StopIteration:
            return None

        node = QuadtreeNode(y, x, h, w, depth)

        if flag == 0: # Split
            half_h, half_w = h // 2, w // 2
            node.children = [
                self._decode_recursive(bit_stream, color_stream, y, x, half_h, half_w, depth+1, mode),           # TL
                self._decode_recursive(bit_stream, color_stream, y, x+half_w, half_h, w-half_w, depth+1, mode),  # TR
                self._decode_recursive(bit_stream, color_stream, y+half_h, x, h-half_h, half_w, depth+1, mode),  # BL
                self._decode_recursive(bit_stream, color_stream, y+half_h, x+half_w, h-half_h, w-half_w, depth+1, mode) # BR
            ]
        else: # Leaf
            self._read_colors_dpcm(node, color_stream, mode)
            
        return node

    # --- Lógica de Color con DPCM (Delta Coding) ---

    def _write_colors_dpcm(self, node, stream: bytearray, mode: str):
        # Obtener colores (relleno negro si es None)
        raw = [node.color_tl, node.color_tr, node.color_bl, node.color_br]
        colors = [c if c is not None else np.zeros(3, dtype=np.float32) for c in raw]

        if mode == 'optimized':
            # --- CORRECCIÓN: CLOSED-LOOP DPCM ---
            ys, cbs, crs = [], [], []
            for col in colors:
                y, cb, cr = self._rgb_to_ycbcr(col)
                ys.append(y)
                cbs.append(cb)
                crs.append(cr)
            
            # 1. Luma (Y)
            for val_y in ys:
                diff = int(val_y - self.prev_y) % 256
                stream.append(diff)
                
                # ¡CRÍTICO! Actualizamos el predictor con lo que ve el DECODER, no el original
                reconstructed_val = (self.prev_y + diff) % 256
                self.prev_y = reconstructed_val 

            # 2. Chroma (CbCr)
            avg_cb = np.mean(cbs)
            avg_cr = np.mean(crs)
            
            diff_cb = int(avg_cb - self.prev_cb) % 256
            diff_cr = int(avg_cr - self.prev_cr) % 256
            
            stream.append(diff_cb)
            stream.append(diff_cr)
            
            # ¡CRÍTICO! Actualizamos con el valor reconstruido
            self.prev_cb = (self.prev_cb + diff_cb) % 256
            self.prev_cr = (self.prev_cr + diff_cr) % 256

        elif mode == 'gradient':
            for col in colors:
                # Convertimos a entero para asegurar consistencia
                col_int = np.clip(col, 0, 255).astype(np.int32)
                prev_int = np.clip(self.prev_dc, 0, 255).astype(np.int32)
                
                diff = (col_int - prev_int) % 256
                stream.extend(diff.astype(np.uint8).tobytes())
                
                # Actualizar con reconstruido
                reconstructed = (prev_int + diff) % 256
                self.prev_dc = reconstructed.astype(np.float32)

        elif mode == 'flat':
            avg = np.mean(colors, axis=0)
            avg_int = np.clip(avg, 0, 255).astype(np.int32)
            prev_int = np.clip(self.prev_dc, 0, 255).astype(np.int32)
            
            diff = (avg_int - prev_int) % 256
            stream.extend(diff.astype(np.uint8).tobytes())
            
            reconstructed = (prev_int + diff) % 256
            self.prev_dc = reconstructed.astype(np.float32)

    def _read_colors_dpcm(self, node, stream, mode: str):
        def read_n(n):
            return [next(stream) for _ in range(n)]

        if mode == 'optimized':
            # --- DECODIFICACIÓN YCbCr DPCM ---
            # 1. Recuperar Luma (4 valores)
            y_deltas = read_n(4)
            rec_ys = []
            for d in y_deltas:
                val = (self.prev_y + d) % 256
                rec_ys.append(val)
                self.prev_y = val
            
            # 2. Recuperar Chroma (2 valores)
            d_cb, d_cr = read_n(2)
            cb = (self.prev_cb + d_cb) % 256
            cr = (self.prev_cr + d_cr) % 256
            self.prev_cb, self.prev_cr = cb, cr
            
            # 3. Reconstrucción Final
            node.color_tl = self._ycbcr_to_rgb(rec_ys[0], cb, cr)
            node.color_tr = self._ycbcr_to_rgb(rec_ys[1], cb, cr)
            node.color_bl = self._ycbcr_to_rgb(rec_ys[2], cb, cr)
            node.color_br = self._ycbcr_to_rgb(rec_ys[3], cb, cr)

        elif mode == 'gradient':
            rec_colors = []
            for _ in range(4):
                diff = np.array(read_n(3), dtype=np.float32)
                val = (self.prev_dc + diff) % 256
                rec_colors.append(val)
                self.prev_dc = val
            node.color_tl, node.color_tr, node.color_bl, node.color_br = rec_colors

        elif mode == 'flat':
            diff = np.array(read_n(3), dtype=np.float32)
            val = (self.prev_dc + diff) % 256
            self.prev_dc = val
            node.color_tl = node.color_tr = node.color_bl = node.color_br = val

    # --- Utilidades de Bits (Low Level Engineering) ---

    def _pack_bits(self, bits: list) -> bytes:
        """Empaqueta una lista de [0,1,1,0...] en bytes reales."""
        byte_arr = bytearray()
        current_byte = 0
        bit_count = 0
        
        for b in bits:
            if b: current_byte |= (1 << (7 - bit_count))
            bit_count += 1
            if bit_count == 8:
                byte_arr.append(current_byte)
                current_byte = 0
                bit_count = 0
        
        if bit_count > 0: byte_arr.append(current_byte)
        return bytes(byte_arr)

    def _bit_generator(self, packed_bytes):
        """Generador bit a bit."""
        for byte in packed_bytes:
            for i in range(7, -1, -1):
                yield (byte >> i) & 1

    # --- Utilidades Matemáticas ---

    def _rgb_to_ycbcr(self, col):
        r, g, b = col
        y  =  0.299*r + 0.587*g + 0.114*b
        cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
        cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
        return y, cb, cr

    def _ycbcr_to_rgb(self, y, cb, cr):
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return np.array([r, g, b], dtype=np.float32)