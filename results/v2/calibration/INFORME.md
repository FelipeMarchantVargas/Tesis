# Calibración no circular — informe (2026-11-17)

Set: 20 recortes 768×768 de CLIC 2020 *validation* (10 Professional + 10 Mobile), disjunto del benchmark.
Métrica: BD-Rate en LPIPS (AlexNet) frente al baseline; además, mediana del BD-Rate **por imagen** (robusta).

## 1. Baseline justo: T = 0
| T (α = β = 0) | 0 | 2,5 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| BD-Rate LPIPS vs T = 0 | 0 | +0,02 % | +0,9 % | +10,3 % | **+63,4 %** | sin solape |

Con el T = 19,95 de la memoria, el baseline necesita ~63 % más bits que con T = 0. Con T = 0 (árbol completo) la poda RDO es óptima dentro de sus modos, y α deja de tener efecto.

## 2. Optuna (BD-Rate LPIPS, TranSalNet-Res, 40 trials)
Óptimo: T ≈ 0,6, α ≈ 19,7 (irrelevante), β ≈ 20 (borde) → −8,65 % (PSNR +0,56 %). Con los valores de la memoria: −1,28 %.

## 3. Barrido de β (T = 0) — no hay óptimo interior
BD-Rate LPIPS (curva promedio): TSN-Res −7,0 % (β 10) → −13,9 % (β 160) → −17,6 % (β 10⁴); U²-Net se satura en ~−3 %; MobileSal −4,8 % (β 160).
Problema: el solape de las curvas cae del 65 % al 35 % al subir β.

## 4. Normalizar el peso invierte el resultado
Con w = (1 + βS)/(1 + βS̄_img), mismo rango de BPP (solape 85–97 %): TSN-Res pierde en LPIPS para todo β (+0,9 % con β 2 … +8,2 % con β 1000). Por imagen (β 160): sin normalizar, mediana −12,7 % (19/20 mejoran); normalizado, +4,5 % (0/20).

## 5. Control sin saliencia: la ganancia viene del acoplamiento λ→Q
Peso **uniforme** k en toda la imagen (J = kD + λR; Q sigue fijado por λ = 100 − 12 ln(λ+1)):

| k | 2 | 5 | 20 | 80 | 320 | 1280 |
|---|---|---|---|---|---|---|
| BD LPIPS (mediana por imagen) | −5,9 % | −10,0 % | −13,2 % | −14,8 % | **−15,1 %** | −14,0 % |

**Sin usar saliencia** se obtiene tanta o más ganancia que con TranSalNet-Res (−12,7 %). El supuesto efecto de la saliencia es, en realidad, bajar el λ efectivo del RDO mientras la cuantización sigue atada al λ nominal.

## 6. Saliencia normalizada sobre el baseline calibrado (k = 320)
Las 4 redes, β ∈ {1, 4, 16}: BD LPIPS entre −0,01 % y +0,30 %, y mejoran 3–11 de 20 imágenes (ruido). **Sin efecto.**

## 7. Por qué
Incluso con k = 1, entre el 91 % y el 99 % de las hojas son DCT 8×8: el árbol es casi una grilla uniforme. La tasa la controla Q(λ), que es **global**. La saliencia solo puede cambiar la elección DCT ↔ plano/interp en pocos bloques, porque el diseño **no permite variar la cuantización por región**.

## Implicancias
- El −13,52 % de la memoria parece explicarse por (a) un baseline debilitado (T alto) y (b) el acoplamiento λ→Q, no por la guía semántica.
- Con este códec no hay parámetros de saliencia significativos que calibrar.
- Para que la saliencia pueda aportar, debe controlar la **cuantización local** (Q o paso por bloque según S, como el ROI-QP en HEVC/VVC), y λ y Q deben elegirse en conjunto (envolvente de Pareto sobre una grilla (λ, Q)).

Código: `calibrate.py`, `calibrate_beta.py`, `plot_calibration.py`; opciones nuevas en `QuadtreeCompressor.compress` (`normalize_weight`, `weight_scale`, ambas desactivadas por defecto). Datos: `perimg_*.csv`, `beta_*`, `baseline_*`.

## 8. Experimento A — envolvente de Pareto (λ × Q desacoplados) — `results/v2/hull/`
Grilla λ ∈ {0,1 … 1250} × Q ∈ {10 … 90}; frente de Pareto por imagen; BD-Rate (mediana por imagen) frente al baseline (β = 0).

| Config (peso normalizado) | LPIPS | mejoran | PSNR |
|---|---|---|---|
| TranSalNet-Res β = 4 | +0,54 % | 7/20 | +0,85 % |
| TranSalNet-Res β = 16 | +0,78 % | 6/20 | +2,00 % |
| U²-Net β = 16 | −0,05 % | 11/20 | +1,77 % |

→ Con λ y Q desacoplados, ponderar el RDO con saliencia **no mejora** LPIPS y cuesta PSNR.

## 9. Experimento B — cuantización local guiada por saliencia — `results/v2/local_q/`
Mapa de 32×32 celdas, 4 niveles, en el bitstream; tabla DCT × 2^(η(0,5 − ℓ)); RDO MSE (β = 0). Mismo protocolo que A.

| Config | LPIPS | mejoran | PSNR |
|---|---|---|---|
| TranSalNet-Res η = 1 | +1,01 % | 1/20 | +0,81 % |
| TranSalNet-Res η = 2 | +2,35 % | 1/20 | +1,15 % |
| TranSalNet-Res η = 3 | +4,19 % | 3/20 | +1,75 % |
| U²-Net η = 2 | +0,97 % | 9/20 | +1,84 % |
| control: sesgo central η = 2 | +1,86 % | 5/20 | +3,05 % |
| control: mapa aleatorio η = 2 | +2,02 % | 0/20 | +2,25 % |

→ Toda cuantización no uniforme **empeora** LPIPS y PSNR, y más cuanto mayor es η. La saliencia apenas se distingue de los controles (TSN-Res η = 2 ≈ aleatorio). Con métricas globales (LPIPS, PSNR), la cuantización uniforme ya es la mejor, como predice la teoría de asignación de bits para MSE.

## Conclusión
Con métricas de referencia completa globales, ni la versión original (peso en el RDO) ni el rediseño B (cuantización local) muestran beneficio de la saliencia una vez que el baseline y el acoplamiento λ→Q están controlados. Demostrar la hipótesis requiere una evaluación que mida la calidad *donde se mira*, sin circularidad: mapas de fijación **reales** (eye-tracking, p. ej. MIT1003/SALICON/CAT2000) como pesos de la métrica, o un test subjetivo (2AFC/MOS).
