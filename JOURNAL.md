# Bitácora de Desarrollo - Tesis Quadtree-CNN

**Autor:** Felipe André Marchant Vargas
**Período:** Noviembre 2025 - Presente

Este documento sirve como un registro central de las decisiones de diseño, notas técnicas, resultados de experimentos y desafíos encontrados durante el desarrollo del proyecto de memoria de título.

---

## 1. Arquitectura y Decisiones de Herramientas

- **Lenguaje:** Python 3.13.
- **Entorno Virtual:** Se utiliza `venv` para aislar las dependencias del proyecto y asegurar la reproducibilidad.
- **Control de Versiones:** Git, con el repositorio alojado en GitHub.
- **Librerías Principales:**
  - **OpenCV (`cv2`):** Utilizada para todas las operaciones básicas de imagen (lectura, escritura, redimensionamiento, conversiones de color, dibujo).
  - **NumPy:** La base para la manipulación de imágenes como matrices numéricas. Es el formato de datos que OpenCV utiliza internamente.
  - **PyTorch:** Framework de Deep Learning seleccionado para cargar y ejecutar el modelo de Red Neuronal Convolucional pre-entrenado.
  - **Matplotlib:** Librería de visualización. Inicialmente usada para mostrar resultados, pero se descartó para la salida final de imágenes debido a problemas de backend gráfico en el entorno de desarrollo.

---

## 2. Notas Técnicas Clave

### a) Manejo de Imágenes: OpenCV vs. PyTorch

Uno de los primeros descubrimientos técnicos fue la diferencia en cómo OpenCV y PyTorch manejan los datos de imagen.

- **Formato de Color de OpenCV:**

  - OpenCV lee las imágenes por defecto en formato **BGR** (Azul, Verde, Rojo).
  - **Implicación Crítica:** La mayoría de las otras librerías y modelos pre-entrenados (incluido el que usaremos) esperan el formato estándar **RGB**. Es **obligatorio** convertir la imagen antes de pasarla a la CNN o a Matplotlib.
  - **Código:** `imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)`

- **Estructura de Datos (Dimensiones):**
  - **OpenCV/NumPy:** Representa las imágenes como `(Altura, Ancho, Canales)`, es decir, `(H, W, C)`.
  - **PyTorch:** Espera los tensores de imagen en el formato `(Canales, Altura, Ancho)`, es decir, `(C, H, W)`.
  - **Implicación Crítica:** Es necesario permutar (reordenar) las dimensiones del array de NumPy antes de convertirlo en un tensor para PyTorch.
  - **Código:** `tensor_pytorch = torch.from_numpy(imagen_rgb).permute(2, 0, 1)`

### b) Implementación del Quadtree Tradicional (Línea Base)

La primera versión del compresor se basa en principios clásicos de la literatura:

- **Criterio de Subdivisión:** Se utiliza la **desviación estándar media** de los tres canales de color (B, G, R) como métrica de "error" o "detalle". Un valor alto indica una región compleja que necesita ser subdividida.
- **Condiciones de Parada:** La recursión se detiene si:
  1.  La desviación estándar de la región es **menor** a un `variance_threshold` definido.
  2.  Se alcanza la profundidad máxima (`max_depth`).
  3.  El tamaño del cuadrante es menor o igual a un `min_size`.
- **Representación de Hojas:** Los nodos hoja (que no se subdividen) se representan por el **color promedio** de su región.

---

## 3. Registro de Desafíos y Soluciones (Troubleshooting)

- **Problema:** `ModuleNotFoundError: No module named pip` dentro del `venv` en la terminal de VS Code.

  - **Diagnóstico:** El entorno virtual (`venv`) estaba corrupto.
  - **Solución:** Eliminar la carpeta `venv` (`rm -rf venv`) y recrearla desde cero (`python3 -m venv venv`), para luego reinstalar las dependencias con `pip install -r requirements.txt`.

- **Problema:** `OSError: [Errno 28] No queda espacio en el dispositivo` durante la instalación de PyTorch.

  - **Diagnóstico Falso:** El disco duro principal tenía espacio de sobra.
  - **Diagnóstico Real:** El directorio temporal del sistema (`/tmp`), que `pip` usa para las descargas, estaba lleno o era una partición de RAM (tmpfs) demasiado pequeña para el paquete de 700MB.
  - **Solución:** Forzar a `pip` a usar un directorio temporal local con el comando: `TMPDIR=./pip_temp pip install -r requirements.txt`.

- **Problema:** `AttributeError: module 'gi' has no attribute 'require_version'` al intentar usar `matplotlib.pyplot`.
  - **Diagnóstico:** Problema con el backend gráfico por defecto de Matplotlib (`GTK4`) en el sistema. El error ocurría en la creación de la figura (`plt.subplots`), no al mostrarla.
  - **Solución:** Abandonar Matplotlib para la tarea de guardar la imagen de comparación. Se reimplementó la visualización usando únicamente **OpenCV y NumPy** (`np.hstack` para unir las imágenes y `cv2.imwrite` para guardarlas), lo cual es más robusto y no depende de backends de GUI.

---

## 4. Próximos Pasos

- [ ] **Implementar el Módulo de Saliency:** Crear `src/saliency.py` para cargar el modelo U²-Net y generar mapas de prominencia.
- [ ] **Integrar Saliency en el Quadtree:** Modificar la clase `QuadtreeCompressor` para usar un criterio de subdivisión híbrido (varianza de color + prominencia).
- [ ] **Realizar primera comparación visual:** Generar una imagen que muestre el resultado del Quadtree tradicional vs. el Quadtree-CNN.
