# Compresión de Imágenes mediante Quadtrees Guiados por Redes Neuronales Convolucionales

**Trabajo de Título para Ingeniería Civil Informática**  
**Autor:** Felipe André Marchant Vargas  
**Profesor Guía:** Roberto León, PhD. Computer Science
**Profesor Co-Guía:** Jorge Díaz, MSc. Computer Science
**Universidad:** Universidad Técnica Federico Santa María

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📖 Resumen

Este proyecto aborda el problema de la compresión de imágenes con pérdida, buscando optimizar la calidad perceptual en lugar de métricas puramente matemáticas. Los métodos tradicionales como JPEG degradan la calidad de manera uniforme, mientras que los algoritmos de Quadtree estándar carecen de entendimiento semántico.

Esta memoria de título propone el diseño y la evaluación de un algoritmo de compresión híbrido que utiliza un modelo de Red Neuronal Convolucional (CNN) para generar un **mapa de prominencia visual** (_saliency map_). Este mapa guía el proceso de subdivisión adaptativa de un Quadtree, asignando mayor detalle y profundidad a las regiones de interés semántico, y aplicando una compresión más agresiva en las zonas menos relevantes para la percepción humana.

## 🎯 Objetivos del Proyecto

### Objetivo General

Diseñar y evaluar un algoritmo de compresión de imágenes basado en la subdivisión adaptativa de Quadtrees, utilizando un mapa de prominencia visual generado por una Red Neuronal Convolucional para mejorar la calidad perceptual frente a métodos estándar.

### Objetivos Específicos

1.  **Desarrollar el componente base** que permita el cálculo de la prominencia de imágenes a partir de una CNN pre-entrenada.
2.  **Implementar un algoritmo de compresión híbrido** basado en Quadtree y guiado por la prominencia visual.
3.  **Evaluar el algoritmo propuesto** en base a la calidad perceptual (usando métricas como SSIM y LPIPS) y compararlo con métodos estándar como JPEG y Quadtree tradicional a tasas de compresión equivalentes.

## 🛠️ Instalación y Configuración del Entorno

Sigue estos pasos para configurar el entorno de desarrollo en un sistema basado en Debian/Ubuntu (como Pop!\_OS).

**1. Clonar el repositorio:**

```bash
git clone [URL-DE-TU-REPOSITORIO]
cd [NOMBRE-DEL-REPOSITORIO]
```

**2. Crear y activar un entorno virtual de Python:**

```bash
python3 -m venv venv
source venv/bin/activate
```

_Para desactivar el entorno, simplemente ejecuta `deactivate`._

**3. Instalar las dependencias:**
Todas las bibliotecas necesarias están listadas en `requirements.txt`.

```bash
pip install -r requirements.txt
```

_(Nota: Asegúrate de tener instalados los drivers de NVIDIA y el CUDA Toolkit si vas a usar la GPU)._

## 🚀 Uso

Aquí se detallan los comandos para ejecutar los procesos principales del proyecto.

**1. Comprimir una imagen:**

```bash
python main.py compress \
    --input path/to/your/image.jpg \
    --output path/to/compressed_file.qt \
    --model path/to/saliency_model.pth \
    --threshold 0.95
```

**2. Descomprimir una imagen:**

```bash
python main.py decompress \
    --input path/to/compressed_file.qt \
    --output path/to/reconstructed_image.png
```

**3. Ejecutar la evaluación de métricas:**

```bash
python evaluate.py \
    --dataset path/to/image_dataset/ \
    --methods jpeg cnn_quadtree traditional_quadtree \
    --output results/evaluation.csv
```

## 📂 Estructura del Repositorio

```
.
├── data/                  # Contiene los datasets de imágenes para entrenamiento y prueba.
├── notebooks/             # Jupyter notebooks para experimentación, análisis y visualización.
├── results/               # Almacena las imágenes de salida, gráficos y reportes de métricas.
├── src/                   # Código fuente principal del proyecto.
│   ├── compression.py     # Lógica de compresión y descompresión con Quadtree.
│   ├── model.py           # Definición del modelo de CNN para prominencia visual.
│   ├── utils.py           # Funciones de ayuda (cálculo de métricas, I/O de imágenes).
│   └── ...
├── main.py                # Script principal para ejecutar la compresión/descompresión.
├── evaluate.py            # Script para correr las evaluaciones de rendimiento.
├── requirements.txt       # Lista de dependencias de Python para `pip`.
└── README.md              # Este archivo.
```

## 🧠 Metodología Propuesta

El algoritmo sigue un enfoque de **Muestreo Adaptativo** y **Reconstrucción Continua**:

1.  **Análisis Semántico:** La imagen entra a una CNN que genera un mapa de calor (0.0 a 1.0), indicando qué píxeles son perceptualmente relevantes.
2.  **Muestreo Híbrido:** Se construye un Quadtree. La decisión de dividir un cuadrante depende de una función de costo que combina la varianza del color y la prominencia media.

    - Zonas Importantes: Umbral de error bajo → Alta densidad de nodos (malla fina).

    - Fondos: Umbral de error alto → Baja densidad (malla gruesa).

3.  **Restricción Topológica:** Se aplica una regla de balanceo para asegurar que ningún nodo tenga una diferencia de nivel mayor a 1 con sus vecinos, evitando grietas en la reconstrucción.
4.  **Reconstrucción Continua:** En lugar de pintar bloques sólidos, se utiliza Interpolación Bilineal basada en los vértices del Quadtree. Esto genera una imagen suave y libre de artefactos de bloque ($C^0$ continuity).

## 📄 Licencia

Este proyecto se distribuye bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
