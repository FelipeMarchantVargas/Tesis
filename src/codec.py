import numpy as np
import zlib
import struct
from src.quadtree import QuadtreeNode

class QuadtreeCodec:
    """
    Codec Multi-Modo (Flat / Interp / DCT) con Cabecera de Calidad.
    """

    def compress(self, root: QuadtreeNode, shape: tuple, quality_factor: int = 50) -> bytes:
        h, w = shape
        
        self.prev_y  = 0.0
        self.prev_cb = 128.0
        self.prev_cr = 128.0
        self.prev_dc = 0

        structure_bits = []
        payload_data = bytearray()
        
        self._encode_recursive(root, structure_bits, payload_data)
        packed_structure = self._pack_bits(structure_bits)
        
        # Header: H(2), W(2), Ver(1), Quality(1)
        header = struct.pack('>HHBB', h, w, 3, int(quality_factor))
        
        struct_len = struct.pack('>I', len(packed_structure))
        full_payload = header + struct_len + packed_structure + payload_data
        return zlib.compress(full_payload, level=9)

    def decompress(self, compressed_data: bytes):
        try:
            full_payload = zlib.decompress(compressed_data)
        except:
            raise ValueError("Zlib Error.")

        self.prev_y  = 0.0
        self.prev_cb = 128.0
        self.prev_cr = 128.0
        self.prev_dc = 0

        ptr = 0
        h, w, ver, quality = struct.unpack('>HHBB', full_payload[ptr:ptr+6])
        ptr += 6
        
        struct_len = struct.unpack('>I', full_payload[ptr:ptr+4])[0]
        ptr += 4
        
        packed_structure = full_payload[ptr : ptr + struct_len]
        ptr += struct_len
        raw_data = full_payload[ptr:]
        
        bit_stream = self._bit_generator(packed_structure)
        self.data_ptr = 0
        self.data_buffer = raw_data
        
        root = self._decode_recursive(bit_stream, y=0, x=0, h=h, w=w, depth=0)
        
        return root, (h, w), quality

    def _encode_recursive(self, node: QuadtreeNode, bits: list, payload: bytearray):
        if not node.is_leaf:
            bits.append(0)
            for child in node.children: self._encode_recursive(child, bits, payload)
        else:
            bits.append(1)
            if node.mode == 'flat':
                bits.append(0); bits.append(0)
                self._write_flat_dpcm(node, payload)
            elif node.mode == 'interp':
                bits.append(0); bits.append(1)
                self._write_interp_dpcm(node, payload)
            elif node.mode == 'dct':
                bits.append(1); bits.append(0)
                self._write_dct_block(node, payload)
            else:
                bits.append(0); bits.append(0)
                self._write_flat_dpcm(node, payload)

    def _decode_recursive(self, bit_stream, y, x, h, w, depth):
        try: flag = next(bit_stream)
        except StopIteration: return None
        node = QuadtreeNode(y, x, h, w, depth)
        if flag == 0:
            h2, w2 = h // 2, w // 2
            node.children = [
                self._decode_recursive(bit_stream, y, x, h2, w2, depth+1),
                self._decode_recursive(bit_stream, y, x+w2, h2, w2, depth+1),
                self._decode_recursive(bit_stream, y+h2, x, h2, w2, depth+1),
                self._decode_recursive(bit_stream, y+h2, x+w2, h2, w2, depth+1)
            ]
        else:
            m1 = next(bit_stream)
            m2 = next(bit_stream)
            if m1 == 0 and m2 == 0:
                node.mode = 'flat'; self._read_flat_dpcm(node)
            elif m1 == 0 and m2 == 1:
                node.mode = 'interp'; self._read_interp_dpcm(node)
            elif m1 == 1 and m2 == 0:
                node.mode = 'dct'; self._read_dct_block(node)
            else:
                node.mode = 'flat'; self._read_flat_dpcm(node)
        return node

    def _write_flat_dpcm(self, node, payload):
        col = node.color_tl if node.color_tl is not None else np.array([128,128,128])
        y, cb, cr = self._rgb_to_ycbcr(col)
        payload.append(int(y - self.prev_y) % 256); self.prev_y = (self.prev_y + int(y - self.prev_y)) % 256
        payload.append(int(cb - self.prev_cb) % 256); self.prev_cb = (self.prev_cb + int(cb - self.prev_cb)) % 256
        payload.append(int(cr - self.prev_cr) % 256); self.prev_cr = (self.prev_cr + int(cr - self.prev_cr)) % 256

    def _read_flat_dpcm(self, node):
        chunk = self._read_bytes(3)
        y = (self.prev_y + chunk[0]) % 256; self.prev_y = y
        cb = (self.prev_cb + chunk[1]) % 256; self.prev_cb = cb
        cr = (self.prev_cr + chunk[2]) % 256; self.prev_cr = cr
        node.color_tl = self._ycbcr_to_rgb(y, cb, cr)

    def _write_interp_dpcm(self, node, payload):
        corners = [node.color_tl, node.color_tr, node.color_bl, node.color_br]
        ys, cbs, crs = [], [], []
        for c in corners:
            if c is None: c = np.array([128,128,128])
            _y, _cb, _cr = self._rgb_to_ycbcr(c)
            ys.append(_y); cbs.append(_cb); crs.append(_cr)
        for val_y in ys:
            payload.append(int(val_y - self.prev_y) % 256); self.prev_y = (self.prev_y + int(val_y - self.prev_y)) % 256
        avg_cb = np.mean(cbs); avg_cr = np.mean(crs)
        payload.append(int(avg_cb - self.prev_cb) % 256); self.prev_cb = (self.prev_cb + int(avg_cb - self.prev_cb)) % 256
        payload.append(int(avg_cr - self.prev_cr) % 256); self.prev_cr = (self.prev_cr + int(avg_cr - self.prev_cr)) % 256

    def _read_interp_dpcm(self, node):
        chunk_y = self._read_bytes(4); rec_ys = []
        for d_y in chunk_y:
            y = (self.prev_y + d_y) % 256; rec_ys.append(y); self.prev_y = y
        chunk_c = self._read_bytes(2)
        cb = (self.prev_cb + chunk_c[0]) % 256; self.prev_cb = cb
        cr = (self.prev_cr + chunk_c[1]) % 256; self.prev_cr = cr
        node.color_tl = self._ycbcr_to_rgb(rec_ys[0], cb, cr)
        node.color_tr = self._ycbcr_to_rgb(rec_ys[1], cb, cr)
        node.color_bl = self._ycbcr_to_rgb(rec_ys[2], cb, cr)
        node.color_br = self._ycbcr_to_rgb(rec_ys[3], cb, cr)

    def _write_dct_block(self, node, payload):
        coeffs = node.dct_coeffs
        dc_val = coeffs[0]
        payload.extend(struct.pack('>h', dc_val - self.prev_dc))
        self.prev_dc = dc_val
        run = 0
        for i in range(1, 64):
            val = coeffs[i]
            if val == 0: run += 1
            else:
                payload.append(run)
                payload.extend(struct.pack('>h', val))
                run = 0
        payload.append(0); payload.extend(struct.pack('>h', 0))
        col = node.color_tl if node.color_tl is not None else np.array([128,128,128])
        _, cb, cr = self._rgb_to_ycbcr(col)
        payload.append(int(cb - self.prev_cb) % 256); self.prev_cb = (self.prev_cb + int(cb - self.prev_cb)) % 256
        payload.append(int(cr - self.prev_cr) % 256); self.prev_cr = (self.prev_cr + int(cr - self.prev_cr)) % 256

    def _read_dct_block(self, node):
        chunk_dc = self._read_bytes(2)
        dc_val = self.prev_dc + struct.unpack('>h', chunk_dc)[0]; self.prev_dc = dc_val
        
        # --- CORRECCIÓN CRÍTICA AQUÍ ---
        # Usamos np.int32 para evitar OverflowError si el DPCM se acumula por encima de 32767
        coeffs = np.zeros(64, dtype=np.int32) 
        coeffs[0] = dc_val
        
        idx = 1
        while idx < 64:
            run = self._read_bytes(1)[0]
            val = struct.unpack('>h', self._read_bytes(2))[0]
            if run == 0 and val == 0: break
            idx += run
            if idx < 64: coeffs[idx] = val; idx += 1
        node.dct_coeffs = coeffs
        chunk_c = self._read_bytes(2)
        cb = (self.prev_cb + chunk_c[0]) % 256; self.prev_cb = cb
        cr = (self.prev_cr + chunk_c[1]) % 256; self.prev_cr = cr
        node.color_tl = self._ycbcr_to_rgb(128, cb, cr)

    def _pack_bits(self, bits):
        byte_arr = bytearray()
        cur, cnt = 0, 0
        for b in bits:
            if b: cur |= (1 << (7 - cnt))
            cnt += 1
            if cnt == 8: byte_arr.append(cur); cur, cnt = 0, 0
        if cnt > 0: byte_arr.append(cur)
        return bytes(byte_arr)

    def _bit_generator(self, packed):
        for b in packed:
            for i in range(7, -1, -1): yield (b >> i) & 1

    def _read_bytes(self, count):
        chunk = self.data_buffer[self.data_ptr : self.data_ptr + count]
        self.data_ptr += count
        return chunk

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