# Pre-registro — corrida principal con fijaciones reales (MIT1003)

Fijado el 2026-11-18, **antes** de codificar cualquier imagen de la partición de prueba. El commit que
contiene este archivo es la referencia. Protocolo completo: nota «Protocolo de evaluación con fijaciones
reales» en la bóveda de la memoria. Piloto (calibración, 16 imágenes): `bd_summary_calib.csv`.

## Datos
- MIT1003, partición de prueba (803 imágenes, semilla 42; `data/mit1003/derived/split.csv`).
- Submuestra de **300 imágenes** al azar (semilla 42): lista fija en `test_images_300.txt`.
- G: fijaciones de los 15 observadores sin la primera, gaussiana σ = 34 px (1°), suma 1
  (`config.json`; CC mediana 0,948 contra los mapas oficiales).

## Codificación
- Cuantización local (qmap 32×32, 4 niveles), RDO MSE (β = 0), mapa guía normalizado por rangos por celda.
- **η = 2 común para todos los brazos** (decidido tras el piloto: la ganancia crece monótonamente con η
  en todos los brazos útiles y el costo en el fondo también; un η común quita el ajuste por brazo).
- Brazos: base (η = 0), TranSalNet-Res, TranSalNet-Dense, U²-Net, MobileSal, sesgo central empírico
  (calibración), aleatorio, oráculo A, oráculo B.
- Grilla λ ∈ {0,1; 0,5; 2; 10; 50; 250; 1250} × Q ∈ {10, 20, 30, 45, 60, 75, 90}.

## BD-Rate
- Por imagen: frente de Pareto de cada brazo y del brazo de referencia sobre la métrica evaluada;
  `bd_rate_interp` (experiment_hull.py) en el rango de calidad común.
- Se excluye la imagen si el solape de calidad es < **50 %** del rango del brazo de referencia; se informa
  cuántas se excluyen por comparación.
- Evaluación: G de los 15 observadores, salvo el oráculo A (G de la mitad B) y el oráculo B (G de la mitad A).

## Hipótesis confirmatorias (métrica primaria: FW-LPIPS)
| | Estadístico por imagen | Hipótesis |
|---|---|---|
| H0 | promedio de BD(oráculo A vs. base, G_B) y BD(oráculo B vs. base, G_A) | mediana < 0 |
| H1 | BD(TranSalNet-Res vs. base, G) | mediana < 0 |
| H2 | BD(TranSalNet-Res vs. **sesgo central**, G) — comparación directa, el sesgo central es la referencia | mediana < 0 |

- Prueba de Wilcoxon de rangos con signo, unilateral (alternativa: menor que 0), α = 0,05, corrección de
  Holm sobre H0–H2.
- Mediana con IC 95 % por *bootstrap* de imágenes (10 000 remuestreos, semilla 42) y conteo «mejoran k/N».
- **Efecto mínimo relevante: mediana ≤ −2 %.** Significativo pero por encima de −2 % → «sin efecto práctico».
- Compuerta: si H0 no se rechaza, H1 y H2 se informan pero no se interpretan como evidencia sobre la saliencia.

## Descriptivo (sin prueba)
- H3, costo: BG-LPIPS, LPIPS global y PSNR global de TranSalNet-Res vs. base (mediana e IC).
- Resto de brazos y métricas (FW-PSNR, FW-SSIM), y H2 con TranSalNet-Dense: exploratorio.
- Sensibilidad con η = 3: opcional y posterior, no confirmatoria.

## Desviaciones respecto del protocolo original (declaradas)
- η común fijado tras el piloto, en lugar del mejor η por brazo (la regla original llevaba η al borde).
- 300 de las 803 imágenes de prueba, por costo (≈ 16 h).
- LPIPS calculado en GPU (diferencia relativa < 1e-6 frente a CPU).
- Piloto con 16 imágenes en vez de 30 (por costo).
- El código de las pruebas H0–H2 se escribe después de este registro, siguiendo exactamente este texto.
