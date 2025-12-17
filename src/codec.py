import numpy as np
import zlib
import struct
from src.quadtree import QuadtreeNode

class QuadtreeCodec:
    """
    Codec Multi-Modo (Multi-Mode Prediction RDO).
    Estructura Bitstream:
    - Split Node: 0
    - Leaf Node: 1 + [Mode Bit]
        - Mode 0 (Flat): Data (1 Y + CbCr)
        - Mode 1 (Interp): Data (4 Y + CbCr)
    """

    def compress(self, root: QuadtreeNode, shape: tuple, mode: str = 'optimized') -> bytes:
        h, w = shape
        
        # Reset Predictors (Closed-Loop)
        self.prev_y  = 0.0
        self.prev_cb = 128.0
        self.prev_cr = 128.0

        structure_bits = []
        color_data = bytearray()
        
        # Encode
        self._encode_recursive(root, structure_bits, color_data)
        
        packed_structure = self._pack_bits(structure_bits)
        
        # Payload
        header = struct.pack('>HHB', h, w, 2) # Mode 2 = Optimized Multi-Mode
        struct_len = struct.pack('>I', len(packed_structure))
        
        full_payload = header + struct_len + packed_structure + color_data
        return zlib.compress(full_payload, level=9)

    def decompress(self, compressed_data: bytes):
        try:
            full_payload = zlib.decompress(compressed_data)
        except:
            raise ValueError("Zlib Error")

        # Reset Predictors
        self.prev_y  = 0.0
        self.prev_cb = 128.0
        self.prev_cr = 128.0

        ptr = 0
        h, w, _ = struct.unpack('>HHB', full_payload[ptr:ptr+5])
        ptr += 5
        
        struct_len = struct.unpack('>I', full_payload[ptr:ptr+4])[0]
        ptr += 4
        
        packed_structure = full_payload[ptr : ptr + struct_len]
        ptr += struct_len
        raw_colors = full_payload[ptr:]
        
        bit_stream = self._bit_generator(packed_structure)
        color_stream = iter(raw_colors)
        
        root = self._decode_recursive(bit_stream, color_stream, 0, 0, h, w, 0)
        return root, (h, w)

    def _encode_recursive(self, node: QuadtreeNode, bits: list, colors: bytearray):
        if not node.is_leaf:
            bits.append(0) # Flag: Split
            for child in node.children:
                self._encode_recursive(child, bits, colors)
        else:
            bits.append(1) # Flag: Leaf
            
            # --- NUEVO: Bit de Modo ---
            if node.mode == 'flat':
                bits.append(0) # Mode 0 = Flat
                self._write_flat_dpcm(node, colors)
            else:
                bits.append(1) # Mode 1 = Interp
                self._write_interp_dpcm(node, colors)

    def _decode_recursive(self, bit_stream, color_stream, y, x, h, w, depth):
        try:
            flag = next(bit_stream)
        except StopIteration:
            return None

        node = QuadtreeNode(y, x, h, w, depth)

        if flag == 0: # Split
            half_h, half_w = h // 2, w // 2
            node.children = [
                self._decode_recursive(bit_stream, color_stream, y, x, half_h, half_w, depth+1),
                self._decode_recursive(bit_stream, color_stream, y, x+half_w, half_h, w-half_w, depth+1),
                self._decode_recursive(bit_stream, color_stream, y+half_h, x, h-half_h, half_w, depth+1),
                self._decode_recursive(bit_stream, color_stream, y+half_h, x+half_w, h-half_h, w-half_w, depth+1)
            ]
        else: # Leaf
            # Leer Bit de Modo
            mode_bit = next(bit_stream)
            if mode_bit == 0:
                node.mode = 'flat'
                self._read_flat_dpcm(node, color_stream)
            else:
                node.mode = 'interp'
                self._read_interp_dpcm(node, color_stream)
            
        return node

    # --- WRITERS (Closed-Loop DPCM) ---

    def _write_flat_dpcm(self, node, stream):
        # Flat: Promedio de las esquinas (o la que se tenga)
        # Nota: El RDO calculó costo basado en un color. Usamos promedio de esquinas actuales.
        raw = [node.color_tl, node.color_tr, node.color_bl, node.color_br]
        # Filtrar Nones
        valid = [c for c in raw if c is not None]
        if not valid: avg_col = np.zeros(3)
        else: avg_col = np.mean(valid, axis=0)

        y, cb, cr = self._rgb_to_ycbcr(avg_col)
        
        # 1. Luma (1 valor)
        diff = int(y - self.prev_y) % 256
        stream.append(diff)
        self.prev_y = (self.prev_y + diff) % 256 # Update Closed-Loop

        # 2. Chroma (1 valor)
        diff_cb = int(cb - self.prev_cb) % 256
        diff_cr = int(cr - self.prev_cr) % 256
        stream.append(diff_cb)
        stream.append(diff_cr)
        self.prev_cb = (self.prev_cb + diff_cb) % 256
        self.prev_cr = (self.prev_cr + diff_cr) % 256

    def _write_interp_dpcm(self, node, stream):
        raw = [node.color_tl, node.color_tr, node.color_bl, node.color_br]
        colors = [c if c is not None else np.zeros(3) for c in raw]
        
        ys, cbs, crs = [], [], []
        for c in colors:
            _y, _cb, _cr = self._rgb_to_ycbcr(c)
            ys.append(_y); cbs.append(_cb); crs.append(_cr)
            
        # 1. Luma (4 valores)
        for val_y in ys:
            diff = int(val_y - self.prev_y) % 256
            stream.append(diff)
            self.prev_y = (self.prev_y + diff) % 256 # Update per pixel

        # 2. Chroma (Promedio, 1 valor compartido)
        avg_cb = np.mean(cbs)
        avg_cr = np.mean(crs)
        
        diff_cb = int(avg_cb - self.prev_cb) % 256
        diff_cr = int(avg_cr - self.prev_cr) % 256
        stream.append(diff_cb)
        stream.append(diff_cr)
        
        self.prev_cb = (self.prev_cb + diff_cb) % 256
        self.prev_cr = (self.prev_cr + diff_cr) % 256

    # --- READERS ---

    def _read_flat_dpcm(self, node, stream):
        # 1. Luma
        d_y = next(stream)
        y = (self.prev_y + d_y) % 256
        self.prev_y = y
        
        # 2. Chroma
        d_cb = next(stream); d_cr = next(stream)
        cb = (self.prev_cb + d_cb) % 256
        cr = (self.prev_cr + d_cr) % 256
        self.prev_cb, self.prev_cr = cb, cr
        
        # Reconstruir (Las 4 esquinas iguales)
        col = self._ycbcr_to_rgb(y, cb, cr)
        node.color_tl = node.color_tr = node.color_bl = node.color_br = col

    def _read_interp_dpcm(self, node, stream):
        # 1. Luma (4 valores)
        rec_ys = []
        for _ in range(4):
            d_y = next(stream)
            y = (self.prev_y + d_y) % 256
            rec_ys.append(y)
            self.prev_y = y
            
        # 2. Chroma (1 valor)
        d_cb = next(stream); d_cr = next(stream)
        cb = (self.prev_cb + d_cb) % 256
        cr = (self.prev_cr + d_cr) % 256
        self.prev_cb, self.prev_cr = cb, cr
        
        # Reconstruir
        node.color_tl = self._ycbcr_to_rgb(rec_ys[0], cb, cr)
        node.color_tr = self._ycbcr_to_rgb(rec_ys[1], cb, cr)
        node.color_bl = self._ycbcr_to_rgb(rec_ys[2], cb, cr)
        node.color_br = self._ycbcr_to_rgb(rec_ys[3], cb, cr)

    # --- HELPERS LOW LEVEL ---
    def _pack_bits(self, bits):
        byte_arr = bytearray()
        cur, cnt = 0, 0
        for b in bits:
            if b: cur |= (1 << (7 - cnt))
            cnt += 1
            if cnt == 8:
                byte_arr.append(cur)
                cur, cnt = 0, 0
        if cnt > 0: byte_arr.append(cur)
        return bytes(byte_arr)

    def _bit_generator(self, packed):
        for b in packed:
            for i in range(7, -1, -1):
                yield (b >> i) & 1

    def _rgb_to_ycbcr(self, c):
        r, g, b = c
        y  =  0.299*r + 0.587*g + 0.114*b
        cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
        cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
        return y, cb, cr

    def _ycbcr_to_rgb(self, y, cb, cr):
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return np.array([r, g, b], dtype=np.float32)