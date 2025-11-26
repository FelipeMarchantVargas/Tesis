# Bitácora de Desarrollo - Tesis Quadtree-CNN

**Autor:** Felipe André Marchant Vargas
**Período:** Noviembre 2025 - Presente

Este documento sirve como un registro central de las decisiones de diseño, notas técnicas, resultados de experimentos y desafíos encontrados durante el desarrollo del proyecto de memoria de título.

---

## 1. Arquitectura y Decisiones de Herramientas

- **Lenguaje:** Python 3.13.
- **Entorno Virtual:** Se utiliza `venv` para aislar las dependencias del proyecto.
- **Control de Versiones:** Git + GitHub.
- **Librerías Principales:**
  - **OpenCV (`cv2`):** Operaciones básicas (lectura, resize, dibujo de primitivas).
  - **NumPy:** Manipulación matricial de imágenes.
  - **PyTorch:** Framework para el modelo de Saliency (U²-Net).
  - **Matplotlib:** Descartada para renderizado final por problemas de backend; se optó por OpenCV puro.

---

## 2. Notas Técnicas Clave

### a) Manejo de Imágenes: OpenCV vs. PyTorch

- **Formato de Color:** OpenCV usa **BGR**. Es mandatorio convertir a **RGB** antes de pasar la imagen a la CNN (`cv2.cvtColor`).
- **Dimensiones:** OpenCV usa `(H, W, C)`, PyTorch requiere `(C, H, W)`. Se requiere permutación de ejes (`torch.from_numpy(...).permute(2, 0, 1)`).

### b) Lógica de Subdivisión Híbrida

Se ha evolucionado del criterio puramente estadístico a uno híbrido:

- **Fórmula:** `Umbral_Efectivo = Umbral_Base * (1.0 - (alpha * Importancia))`
- **Funcionamiento:**
  - Si la región tiene alta importancia en el mapa de prominencia (valor cercano a 1.0), el umbral de error baja drásticamente, forzando la subdivisión incluso si la varianza de color es baja.
  - Esto permite concentrar nodos en áreas semánticas (rostros, objetos) y ahorrar nodos en fondos.

### c) Cambio de Metodología de Reconstrucción

Originalmente se planteó usar DCT o bloques de color promedio. Se ha decidido migrar a **Quadtrees Restringidos con Interpolación Bilineal**.

- **Motivo:** Evitar los artefactos de bloque (mosaico) típicos de JPEG y Quadtrees simples.
- **Técnica:** Se forzará el balanceo del árbol (diferencia de nivel máxima de 1 entre vecinos) para permitir una interpolación suave y continua ($C^0$ continuity) sin necesidad de triangular con Delaunay.

---

## 3. Registro de Desafíos y Soluciones (Troubleshooting)

- **Problema:** `TypeError: _subdivide() missing 1 required positional argument: 'current_depth'`

  - **Contexto:** Al integrar el mapa de prominencia, se modificó la firma de `_subdivide` pero no la llamada inicial en `compress()`.
  - **Solución:** Actualizar la llamada raíz a `self._subdivide(self.root, image, saliency_map, 0)`.

- **Problema:** Visualización de la estructura del Quadtree.

  - **Necesidad:** Se requería verificar si la densidad de nodos realmente obedecía al mapa de prominencia.
  - **Solución:** Se implementó el método `visualize_structure()` que dibuja solo los bordes de las hojas. Al superponerlo con la imagen original, se confirmó visualmente la adaptación de densidad.

- **Problema:** `AttributeError: module 'gi' has no attribute 'require_version'` (Matplotlib).
  - **Solución:** Reemplazo de Matplotlib por `cv2.imwrite` y `np.hstack` para guardar comparativas.

---

## 4. Próximos Pasos (Hoja de Ruta Inmediata)

1.  [ ] **Implementar Nodos con 4 Esquinas:** Modificar `QuadtreeNode` para almacenar colores en `top_left`, `top_right`, etc., en lugar de un solo `color` promedio.
2.  [ ] **Implementar Interpolación Bilineal:** Crear la lógica de renderizado que pinte píxeles interpolando valores, eliminando el efecto de "bloques".
3.  [ ] **Integrar U²-Net Real:** Reemplazar el "Mock Saliency" (círculo blanco) por la inferencia real de la red neuronal.

---

## 📅 Bitácora de Desarrollo

**[2025-11-26] - Versión 0.2.0: Validación de Lógica Híbrida**

- **Hito:** Se integró exitosamente la lógica de subdivisión guiada por un mapa de prominencia.
- **Prueba de Concepto:** Se utilizó un mapa de prominencia simulado (círculo blanco en el centro). Las visualizaciones (`debug_structure_overlay.png`) confirmaron que el algoritmo asigna mayor densidad de nodos en la zona "importante" y menor en el fondo, validando la hipótesis central.
- **Corrección:** Se solucionaron errores de paso de argumentos en la recursión.

**[2025-11-24] - Versión 0.1.0: Algoritmo Base**

- Implementación del Quadtree clásico (Criterio de Varianza).
- Visualización básica por bloques promedio.

**[2025-10-15] - Inicio del Proyecto**

- Definición de propuesta y estructura del repositorio.
