# Compresión de Imágenes Perceptual mediante Quadtrees Restringidos Guiados por Semántica

Este repositorio contiene la implementación oficial del sistema de compresión de imágenes desarrollado para el trabajo de título de Ingeniería Civil Informática. El sistema utiliza una estructura de datos **Restricted Quadtree** (balanceado) guiada por mapas de prominencia visual (Saliency Maps) generados mediante **U2-Net**.

## 📋 Características del Sistema

- **Compresión Adaptativa**: Asigna mayor densidad de nodos a regiones semánticamente importantes.
- **Balanceo Geométrico**: Implementa la regla 2:1 (Restricted Quadtree) para asegurar continuidad en la malla.
- **Reconstrucción Vectorizada**: Utiliza interpolación bilineal optimizada con NumPy (sin bloques sólidos) para una recuperación visual suave.
- **Eficiencia**: Cálculo de métricas de error en $O(1)$ utilizando Imágenes Integrales.

## ⚙️ Requisitos e Instalación

### Prerrequisitos

- Python 3.9 o superior.
- Entorno virtual recomendado.

### Instalación de Dependencias

Crea un archivo `requirements.txt` con el siguiente contenido e instálalo:

```txt
numpy
opencv-python
torch
torchvision
```

Ejecuta:

```
pip install -r requirements.txt
```

### Configuración del Modelo (Opcional)

Por defecto, el sistema funciona en Modo Mock (generando un mapa gaussiano sintético). Para usar la inferencia real con Inteligencia Artificial:

1. Descarga los pesos de U2-Net (u2net.pth).

2. Coloca el archivo en la raíz del proyecto o en una carpeta weights/.

3. Asegúrate de tener el archivo de definición del modelo (u2net.py) accesible para el importador (ver src/saliency.py).

## 🚀 Instrucciones de Uso (CLI)

El sistema se maneja a través de main.py utilizando tres comandos principales: compress, reconstruct y visualize.

1. Compresión

Toma una imagen RGB, genera su mapa de saliencia, construye el Quadtree y guarda los datos comprimidos en un archivo binario (.pkl comprimido con zlib).

```
python main.py compress -i <imagen_entrada> -o <archivo_salida.pkl> [opciones]
```

Ejemplo:

```
python main.py compress -i lenna.png -o lenna_compressed.pkl -t 15.0 -a 0.5
```

2. Reconstrucción

Lee el archivo binario, recupera la estructura del árbol y los colores de las esquinas, y reconstruye la imagen mediante interpolación.

```
python main.py reconstruct -i <archivo_entrada.pkl> -o <imagen_salida.png>
```

Ejemplo:

```
python main.py reconstruct -i lenna_compressed.pkl -o lenna_restaurada.png
```

3. Visualización (Wireframe)

Genera una imagen de diagnóstico superponiendo la estructura del Quadtree (bordes verdes) sobre la reconstrucción. Ideal para visualizar cómo el algoritmo prioriza zonas semánticas.

```
python main.py visualize -i <imagen_entrada> -o <imagen_wireframe.png> [opciones]
```

## 🎛️ Parámetros de Ajuste

La calidad y el peso del archivo dependen críticamente de threshold y alpha.

| Parámetro | Flag | Default | Descripción                                                           |
| --------- | ---- | ------- | --------------------------------------------------------------------- |
| Threshold | -t   | 10.0    | Umbral de Error Geométrico (RMSE).                                    |
| Alpha     | -a   | 0.5     | Influencia de la Semántica (0.0−1.0).                                 |
| Model     | -m   | None    | Ruta al archivo .pth. Si se omite, usa el generador sintético (Mock). |

## 📂 Estructura del Proyecto

```

├── src/
│   ├── __init__.py
│   ├── quadtree.py       # Lógica de compresión, balanceo y reconstrucción
│   └── saliency.py       # Interfaz con U2-Net y Modo Mock
├── main.py               # Punto de entrada CLI
├── requirements.txt      # Dependencias
└── README.md             # Instrucciones
```

## 📝 Notas Técnicas

- Balanceo: El proceso de balanceo es automático. Si un nodo vecino difiere en más de 1 nivel de profundidad, el sistema forzará subdivisiones recursivas hasta cumplir la regla.

- Formato de Archivo: Los archivos .pkl guardados son serializaciones binarias de la lista de hojas y metadatos, comprimidos posteriormente con zlib nivel 9.

Autor: Felipe André Marchant Vargas
Universidad: Universidad Técnica Federico Santa María
Tesis: Compresión de Imágenes mediante Quadtrees Guiados por Redes Neuronales Convolucionales
