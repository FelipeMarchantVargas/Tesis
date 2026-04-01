import numpy as np
import zlib
import struct
from src.quadtree import QuadtreeNode

class QuadtreeCodec:
    """
    Codec Optimizado:
    - Uso reducido de wrappers de numpy en bucles internos.
    - Manejo de buffers directos.
    """

    def compress(self, root: QuadtreeNode, shape: tuple, quality_factor: int = 50) -> bytes:
        h, w = shape
        
        # Reset states
        self.prev_y  = 0
        self.prev_cb = 128
        self.prev_cr = 128
        self.prev_dc = 0

        # Estimate size to preallocate if possible, but bytearray is dynamic.
        # Lists for bits are efficient enough in Python.
        structure_bits = []
        payload_data = bytearray()
        
        self._encode_recursive(root, structure_bits, payload_data)
        packed_structure = self._pack_bits(structure_bits)
        
        header = struct.pack('>HHBB', h, w, 3, int(quality_factor))
        struct_len = struct.pack('>I', len(packed_structure))
        
        # Combine efficiently
        full_payload = header + struct_len + packed_structure + payload_data
        return zlib.compress(full_payload, level=9)

    def decompress(self, compressed_data: bytes):
        try:
            full_payload = zlib.decompress(compressed_data)
        except:
            raise ValueError("Zlib Error.")

        self.prev_y  = 0
        self.prev_cb = 128
        self.prev_cr = 128
        self.prev_dc = 0

        ptr = 0
        h, w, ver, quality = struct.unpack('>HHBB', full_payload[ptr:ptr+6])
        ptr += 6
        
        struct_len = struct.unpack('>I', full_payload[ptr:ptr+4])[0]
        ptr += 4
        
        packed_structure = full_payload[ptr : ptr + struct_len]
        ptr += struct_len
        
        # Slice raw_data without copying if possible (memoryview), 
        # but for safety and simplicity in parsing:
        raw_data = full_payload[ptr:]
        
        self.bit_stream_iter = self._bit_generator(packed_structure)
        self.data_ptr = 0
        self.data_buffer = raw_data
        
        root = self._decode_recursive(h=h, w=w, y=0, x=0, depth=0)
        
        return root, (h, w), quality

    def _encode_recursive(self, node: QuadtreeNode, bits: list, payload: bytearray):
        # Optimization: Local variable access is faster than attribute access
        children = node.children
        if children: # Not leaf
            bits.append(0)
            for child in children: 
                self._encode_recursive(child, bits, payload)
        else:
            bits.append(1)
            mode = node.mode
            if mode == 'flat':
                bits.append(0); bits.append(0)
                self._write_flat_dpcm(node, payload)
            elif mode == 'interp':
                bits.append(0); bits.append(1)
                self._write_interp_dpcm(node, payload)
            elif mode == 'dct':
                bits.append(1); bits.append(0)
                self._write_dct_block(node, payload)
            else:
                bits.append(0); bits.append(0)
                self._write_flat_dpcm(node, payload)

    def _decode_recursive(self, y, x, h, w, depth):
        try: 
            flag = next(self.bit_stream_iter)
        except StopIteration: 
            return None
            
        node = QuadtreeNode(y, x, h, w, depth)
        
        if flag == 0:
            h2, w2 = h >> 1, w >> 1 # Bit shift division
            # Explicit unrolling often not needed in Python recursion limit, but clean
            node.children = [
                self._decode_recursive(y, x, h2, w2, depth+1),
                self._decode_recursive(y, x+w2, h2, w2, depth+1),
                self._decode_recursive(y+h2, x, h2, w2, depth+1),
                self._decode_recursive(y+h2, x+w2, h2, w2, depth+1)
            ]
        else:
            m1 = next(self.bit_stream_iter)
            m2 = next(self.bit_stream_iter)
            
            # Using int comparison is slightly faster than string compare
            if m1 == 0 and m2 == 0:
                node.mode = 'flat'; self._read_flat_dpcm(node)
            elif m1 == 0 and m2 == 1:
                node.mode = 'interp'; self._read_interp_dpcm(node)
            elif m1 == 1 and m2 == 0:
                node.mode = 'dct'; self._read_dct_block(node)
            else:
                node.mode = 'flat'; self._read_flat_dpcm(node)
        return node

    # --- Fast Math Helpers (Avoid numpy scalars in tight loops) ---
    def _rgb_to_ycbcr_fast(self, r, g, b):
        y  =  0.299*r + 0.587*g + 0.114*b
        cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
        cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
        return int(y), int(cb), int(cr)

    def _ycbcr_to_rgb_array(self, y, cb, cr):
        # Returns numpy array directly
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return np.array([r, g, b], dtype=np.float32)

    def _write_flat_dpcm(self, node, payload):
        col = node.color_tl if node.color_tl is not None else (128,128,128)
        y, cb, cr = self._rgb_to_ycbcr_fast(col[0], col[1], col[2])
        
        payload.append((y - self.prev_y) & 0xFF)
        self.prev_y = y # No need to mod 256 here if we just store raw value, but for consistency:
        
        payload.append((cb - self.prev_cb) & 0xFF)
        self.prev_cb = cb
        
        payload.append((cr - self.prev_cr) & 0xFF)
        self.prev_cr = cr

    def _read_flat_dpcm(self, node):
        # Read 3 bytes at once
        chunk = self.data_buffer[self.data_ptr : self.data_ptr + 3]
        self.data_ptr += 3
        
        y = (self.prev_y + chunk[0]) & 0xFF; self.prev_y = y
        cb = (self.prev_cb + chunk[1]) & 0xFF; self.prev_cb = cb
        cr = (self.prev_cr + chunk[2]) & 0xFF; self.prev_cr = cr
        node.color_tl = self._ycbcr_to_rgb_array(y, cb, cr)

    def _write_interp_dpcm(self, node, payload):
        # Optimization: Flatten the loop
        cols = [node.color_tl, node.color_tr, node.color_bl, node.color_br]
        cbs_sum, crs_sum = 0, 0
        
        for c in cols:
            if c is None: r,g,b = 128,128,128
            else: r,g,b = c[0], c[1], c[2]
            y, cb, cr = self._rgb_to_ycbcr_fast(r,g,b)
            payload.append((y - self.prev_y) & 0xFF)
            self.prev_y = y
            cbs_sum += cb
            crs_sum += cr
            
        avg_cb = int(cbs_sum / 4)
        avg_cr = int(crs_sum / 4)
        
        payload.append((avg_cb - self.prev_cb) & 0xFF); self.prev_cb = avg_cb
        payload.append((avg_cr - self.prev_cr) & 0xFF); self.prev_cr = avg_cr

    def _read_interp_dpcm(self, node):
        chunk = self.data_buffer[self.data_ptr : self.data_ptr + 6] # 4 Ys + 1 Cb + 1 Cr
        self.data_ptr += 6
        
        ys = []
        for i in range(4):
            y = (self.prev_y + chunk[i]) & 0xFF
            self.prev_y = y
            ys.append(y)
            
        cb = (self.prev_cb + chunk[4]) & 0xFF; self.prev_cb = cb
        cr = (self.prev_cr + chunk[5]) & 0xFF; self.prev_cr = cr
        
        node.color_tl = self._ycbcr_to_rgb_array(ys[0], cb, cr)
        node.color_tr = self._ycbcr_to_rgb_array(ys[1], cb, cr)
        node.color_bl = self._ycbcr_to_rgb_array(ys[2], cb, cr)
        node.color_br = self._ycbcr_to_rgb_array(ys[3], cb, cr)

    def _write_dct_block(self, node, payload):
        coeffs = node.dct_coeffs
        dc_val = coeffs[0]
        # pack_into could be faster for large buffers, but extend + pack is fine for streams
        payload.extend(struct.pack('>h', dc_val - self.prev_dc))
        self.prev_dc = dc_val
        
        # Run-Length Encoding Logic
        # Optimization: Use numpy nonzero if overhead permits, but standard loop is often fine for RLE
        # given the short length (64).
        run = 0
        # Manual loop is faster than generic iterators here
        for i in range(1, 64):
            val = coeffs[i]
            if val == 0:
                run += 1
            else:
                payload.append(run)
                payload.extend(struct.pack('>h', val))
                run = 0
        # EOB
        payload.append(0)
        payload.extend(struct.pack('>h', 0))
        
        # Color average
        cols = [node.color_tl, node.color_tr, node.color_bl, node.color_br]
        for c in cols:
            if c is None: r,g,b = 128,128,128
            else: r,g,b = c[0], c[1], c[2]
            _, cb, cr = self._rgb_to_ycbcr_fast(r,g,b)
            payload.append((cb - self.prev_cb) & 0xFF); self.prev_cb = cb
            payload.append((cr - self.prev_cr) & 0xFF); self.prev_cr = cr

    def _read_dct_block(self, node):
        # DC
        raw_dc = self.data_buffer[self.data_ptr : self.data_ptr+2]
        self.data_ptr += 2
        diff_dc = struct.unpack('>h', raw_dc)[0]
        
        dc_val = self.prev_dc + diff_dc
        self.prev_dc = dc_val
        
        coeffs = np.zeros(64, dtype=np.int32) # int16 is enough for DCT coeffs usually
        coeffs[0] = dc_val
        
        idx = 1
        while idx < 64:
            run = self.data_buffer[self.data_ptr]
            self.data_ptr += 1
            
            raw_val = self.data_buffer[self.data_ptr : self.data_ptr+2]
            self.data_ptr += 2
            val = struct.unpack('>h', raw_val)[0]
            
            if run == 0 and val == 0: break # EOB
            
            idx += run
            if idx < 64:
                coeffs[idx] = val
                idx += 1
                
        node.dct_coeffs = coeffs
        
        chunk_c = self.data_buffer[self.data_ptr : self.data_ptr+8]
        self.data_ptr += 8
        
        cbs, crs = [], []
        for i in range(4):
            cb = (self.prev_cb + chunk_c[i*2]) & 0xFF; self.prev_cb = cb
            cr = (self.prev_cr + chunk_c[i*2 + 1]) & 0xFF; self.prev_cr = cr
            cbs.append(cb); crs.append(cr)
            
        node.color_tl = self._ycbcr_to_rgb_array(128, cbs[0], crs[0])
        node.color_tr = self._ycbcr_to_rgb_array(128, cbs[1], crs[1])
        node.color_bl = self._ycbcr_to_rgb_array(128, cbs[2], crs[2])
        node.color_br = self._ycbcr_to_rgb_array(128, cbs[3], crs[3])

    def _pack_bits(self, bits):
        # Optimization: String/list join then int conversion is sometimes faster than manual loop
        # But bitwise ops are canonical.
        L = len(bits)
        byte_arr = bytearray((L + 7) // 8)
        idx = 0
        for i in range(0, L, 8):
            chunk = bits[i:i+8]
            val = 0
            for b in chunk:
                val = (val << 1) | b
            # If chunk < 8, shift remaining
            if len(chunk) < 8:
                val <<= (8 - len(chunk))
            byte_arr[idx] = val
            idx += 1
        return bytes(byte_arr)

    def _bit_generator(self, packed):
        for b in packed:
            yield (b >> 7) & 1
            yield (b >> 6) & 1
            yield (b >> 5) & 1
            yield (b >> 4) & 1
            yield (b >> 3) & 1
            yield (b >> 2) & 1
            yield (b >> 1) & 1
            yield b & 1