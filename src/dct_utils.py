import numpy as np
import cv2

# --- 1. MATRICES DE CUANTIZACIÓN ESTÁNDAR (JPEG) ---
# Estas matrices están diseñadas psico-visualmente. 
# Los valores bajos (arriba-izq) preservan frecuencias bajas (brillo, formas).
# Los valores altos (abajo-der) eliminan ruido y detalles finos invisibles.

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

# --- 2. ORDEN ZIG-ZAG ---
# Crítico para la compresión. Agrupa los ceros al final del array.
# Convierte una matriz 8x8 en un array de 64 elementos.
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

# Inversa para reconstruir la matriz desde el array plano
UNZIGZAG_INDICES = np.argsort(ZIGZAG_INDICES)

def get_quantization_matrix(quality_factor=50, is_chroma=False):
    """
    Escala la matriz de cuantización según un factor de calidad (tipo JPEG).
    quality_factor: 1 (peor) a 100 (mejor).
    """
    if quality_factor <= 0: quality_factor = 1
    if quality_factor > 100: quality_factor = 100

    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - 2 * quality_factor

    base_table = CHROMINANCE_QUANT_TABLE if is_chroma else LUMINANCE_QUANT_TABLE
    table = np.floor((base_table * scale + 50) / 100)
    
    # El valor mínimo es 1 para evitar división por cero
    table[table < 1] = 1
    return table

def dct_transform_block(block, quant_matrix):
    """
    Aplica DCT -> Cuantización -> ZigZag.
    Entrada: Bloque NxN (float).
    Salida: Array plano de coeficientes cuantizados (enteros).
    """
    # 1. DCT (Tipo II - Ortogonal)
    # Restamos 128 para centrar en cero (si es uint8 0-255)
    # Asumimos que 'block' ya viene centrado o lo centramos aquí si es necesario.
    # En tu caso, si pasas residuos (error), ya están centrados en 0.
    
    # cv2.dct requiere float32 y tamaño par (generalmente 8x8)
    dct_coeffs = cv2.dct(block.astype(np.float32))
    
    # 2. Cuantización
    # Dividimos y redondeamos. Aquí es donde se pierden datos (Lossy).
    # Si el bloque es > 8x8 (ej 16x16), necesitamos escalar la matriz Q.
    # Por simplicidad inicial, usaremos bloques de 8x8 fijos.
    # Si tu quadtree manda 16x16, lo ideal es partirlo en 4 de 8x8 o escalar Q.
    # Para tu tesis: Soporta SOLO bloques 8x8 para DCT por ahora.
    
    h, w = block.shape
    if h != 8 or w != 8:
        # Fallback simple: Resize de la matriz Q al tamaño del bloque
        q_resized = cv2.resize(quant_matrix, (w, h))
    else:
        q_resized = quant_matrix

    quantized = np.round(dct_coeffs / q_resized)
    
    # 3. ZigZag (Solo si es 8x8, si no, aplanamos normal)
    if h == 8 and w == 8:
        flat_coeffs = quantized.ravel()[ZIGZAG_INDICES]
    else:
        flat_coeffs = quantized.ravel()
        
    return flat_coeffs.astype(np.int16)

def idct_reconstruct_block(flat_coeffs, quant_matrix, shape=(8,8)):
    """
    Aplica UnZigZag -> Descuantización -> IDCT.
    """
    h, w = shape
    
    # 1. UnZigZag
    if h == 8 and w == 8:
        quantized_matrix = np.zeros((64,), dtype=np.float32)
        quantized_matrix[ZIGZAG_INDICES] = flat_coeffs
        quantized_matrix = quantized_matrix.reshape((8, 8))
    else:
        quantized_matrix = flat_coeffs.reshape(shape)
        
    # 2. Descuantización
    if h != 8 or w != 8:
        q_resized = cv2.resize(quant_matrix, (w, h))
    else:
        q_resized = quant_matrix
        
    dct_coeffs = quantized_matrix * q_resized
    
    # 3. IDCT (Inversa)
    block = cv2.idct(dct_coeffs)
    return block