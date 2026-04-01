import numpy as np
import cv2

# --- 1. MATRICES DE CUANTIZACIÓN ---
LUMINANCE_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

CHROMINANCE_QUANT_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)

ZIGZAG_INDICES = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 53, 46, 38, 31, 39, 47,
    54, 60, 61, 55, 52, 45, 62, 63
], dtype=np.int32)

UNZIGZAG_INDICES = np.argsort(ZIGZAG_INDICES)

def get_quantization_matrix(quality_factor=50, is_chroma=False):
    if quality_factor <= 0: quality_factor = 1
    if quality_factor > 100: quality_factor = 100
    scale = 5000 / quality_factor if quality_factor < 50 else 200 - 2 * quality_factor
    base_table = CHROMINANCE_QUANT_TABLE if is_chroma else LUMINANCE_QUANT_TABLE
    table = np.floor((base_table * scale + 50) / 100)
    table[table < 1] = 1
    return table

def batch_dct_transform(blocks_8x8, quant_matrix):
    """
    Procesa un tensor de bloques (N, 8, 8) de forma vectorizada.
    Retorna:
      - coeffs_quant: (N, 64) coeficientes zigzag cuantizados (int16)
      - error_sse: (N,) Suma de errores cuadráticos de reconstrucción por bloque
    """
    # 1. DCT Batch (usando OpenCV loop optimizado o implementación manual, 
    # cv2.dct es muy rápido pero solo acepta 1D o 2D, así que usamos reshape inteligente)
    # Sin embargo, para N bloques, un loop simple con cv2.dct suele ser más lento que
    # matmul si N es masivo, pero cv2 está en C++.
    # Truco: Concatenar verticalmente para hacer una sola llamada a cv2.dct si es posible,
    # pero cv2.dct espera tamaños pares. Haremos un map eficiente.
    
    # Opción más compatible y rápida en numpy puro para batching:
    # Usamos float32.
    n_blocks = blocks_8x8.shape[0]
    
    # Implementación vectorizada manual usando np.matmul es posible si pre-calculamos la base DCT,
    # pero para mantener compatibilidad con cv2.dct (que es Type-II ortogonal), iteramos mínimamente
    # o usamos un truco de reshape. OpenCV no soporta n-dim DCT nativamente.
    # Iterar es el cuello de botella. Vamos a usar un loop list comprehension que es rápido en CPython
    # para llamadas a C extensions.
    
    # Pre-allocate
    dct_blocks = np.empty_like(blocks_8x8)
    for i in range(n_blocks):
        dct_blocks[i] = cv2.dct(blocks_8x8[i])

    # 2. Cuantización Vectorizada
    # Broadcasting: (N, 8, 8) / (8, 8)
    quantized = np.round(dct_blocks / quant_matrix)
    
    # 3. ZigZag Vectorizado
    # Reshape a (N, 64) y aplicar índices
    flat_blocks = quantized.reshape(n_blocks, 64)
    zigzag_coeffs = flat_blocks[:, ZIGZAG_INDICES].astype(np.int16)
    
    # 4. Reconstrucción para cálculo de error (RDO)
    # UnZigZag
    rec_flat = np.zeros((n_blocks, 64), dtype=np.float32)
    rec_flat[:, ZIGZAG_INDICES] = zigzag_coeffs
    rec_quant = rec_flat.reshape(n_blocks, 8, 8)
    
    # Descuantización
    rec_dct = rec_quant * quant_matrix
    
    # IDCT Batch
    diffs = np.empty_like(blocks_8x8)
    for i in range(n_blocks):
        rec_block = cv2.idct(rec_dct[i])
        diffs[i] = blocks_8x8[i] - rec_block
        
    # SSE (Sum Squared Error) por bloque
    errors_sse = np.sum(diffs**2, axis=(1, 2))
    
    # Costo de Rate (aprox bits): non-zeros
    non_zeros = np.count_nonzero(zigzag_coeffs, axis=1)
    
    return zigzag_coeffs, errors_sse, non_zeros

def dct_transform_block(block, quant_matrix):
    """Versión legacy para single block"""
    dct_coeffs = cv2.dct(block.astype(np.float32))
    h, w = block.shape
    q_resized = quant_matrix if (h==8 and w==8) else cv2.resize(quant_matrix, (w, h))
    quantized = np.round(dct_coeffs / q_resized)
    if h == 8 and w == 8:
        return quantized.ravel()[ZIGZAG_INDICES].astype(np.int16)
    return quantized.ravel().astype(np.int16)

def idct_reconstruct_block(flat_coeffs, quant_matrix, shape=(8,8)):
    h, w = shape
    if h == 8 and w == 8:
        quantized_matrix = np.zeros((64,), dtype=np.float32)
        quantized_matrix[ZIGZAG_INDICES] = flat_coeffs
        quantized_matrix = quantized_matrix.reshape((8, 8))
        q_resized = quant_matrix
    else:
        quantized_matrix = flat_coeffs.reshape(shape)
        q_resized = cv2.resize(quant_matrix, (w, h))
        
    dct_coeffs = quantized_matrix * q_resized
    return cv2.idct(dct_coeffs)