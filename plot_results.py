import matplotlib
# Configurar backend no interactivo ANTES de importar pyplot
# Esto evita el error de GTK/Qt/Tkinter en servidores o entornos virtuales
matplotlib.use('Agg') 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_all_metrics(csv_path="results/benchmark_results_full_metrics.csv", output_dir="results/plots"):
    # 1. Configuración Inicial
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {csv_path}")
        return

    # Definir estilo y paleta
    sns.set_theme(style="whitegrid")
    
    unique_methods = df['Method'].unique()
    # Usamos una paleta amigable para daltónicos o simplemente clara
    palette = sns.color_palette("bright", len(unique_methods))
    method_colors = dict(zip(unique_methods, palette))

    print(f"Cargado {len(df)} filas. Métodos encontrados: {unique_methods}")

    # -------------------------------------------------------------------------
    # 2. DEFINICIÓN DE GRÁFICOS A GENERAR
    # -------------------------------------------------------------------------
    
    # A. Curvas de Calidad (Rate-Distortion Curves)
    qa_metrics = [
        ('PSNR', 'PSNR (dB)', 'Higher is better'),
        ('SSIM', 'SSIM', 'Higher is better'),
        ('MS-SSIM', 'MS-SSIM', 'Higher is better'),
        ('SW-SSIM', 'Saliency Weighted SSIM', 'Higher is better'),
        ('VIF', 'Visual Information Fidelity', 'Higher is better'),
        ('LPIPS', 'LPIPS (Perceptual Error)', 'Lower is better'),
    ]

    # B. Métricas de Rendimiento
    perf_metrics = [
        ('Enc_Time(s)', 'Encoding Time (s)', 'Lower is better'),
        ('Dec_Time(s)', 'Decoding Time (s)', 'Lower is better'),
        ('Enc_Mem(MB)', 'Encoding Peak Memory (MB)', 'Lower is better'),
        ('Dec_Mem(MB)', 'Decoding Peak Memory (MB)', 'Lower is better'),
    ]

    # -------------------------------------------------------------------------
    # 3. GENERACIÓN DE CURVAS RD (Métricas vs BPP)
    # -------------------------------------------------------------------------
    print("Generando curvas Rate-Distortion...")
    
    # Unimos ambas listas para iterar
    all_metrics_to_plot = qa_metrics + perf_metrics

    for metric, ylabel, direction in all_metrics_to_plot:
        if metric not in df.columns:
            print(f"Advertencia: La métrica '{metric}' no está en el CSV. Saltando.")
            continue

        plt.figure(figsize=(10, 6))
        
        # Agrupamos por Method y Param para obtener el punto medio exacto de cada configuración
        df_mean = df.groupby(['Method', 'Param']).mean(numeric_only=True).reset_index()
        df_mean = df_mean.sort_values(by=['Method', 'BPP'])

        sns.lineplot(
            data=df_mean, 
            x='BPP', 
            y=metric, 
            hue='Method', 
            style='Method',
            markers=True, 
            dashes=False,
            palette=method_colors,
            linewidth=2.5,
            markersize=8
        )

        plt.title(f'{ylabel} vs Bitrate')
        plt.xlabel('Bits Per Pixel (BPP)')
        plt.ylabel(ylabel)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        safe_name = metric.replace("(", "").replace(")", "").replace("/", "_")
        plt.savefig(f"{output_dir}/RD_Curve_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # -------------------------------------------------------------------------
    # 4. GRÁFICOS DE BARRAS (Resumen Global)
    # -------------------------------------------------------------------------
    print("Generando gráficos de barras de resumen...")
    
    summary_metrics = [
        ('Enc_Time(s)', 'Average Encoding Time (s)'),
        ('Dec_Time(s)', 'Average Decoding Time (s)'),
        ('Enc_Mem(MB)', 'Average Peak Memory (MB)'),
        ('CR', 'Average Compression Ratio'),
    ]

    for metric, title in summary_metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        sns.barplot(
            data=df, 
            x='Method', 
            y=metric, 
            hue='Method',
            palette=method_colors,
            capsize=.1,
            errorbar='sd' # Muestra desviación estándar
        )
        
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel("Method")
        plt.xticks(rotation=15)
        
        # Mover la leyenda afuera si molesta, o quitarla si es redundante con el eje X
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        safe_name = metric.replace("(", "").replace(")", "").replace("/", "_")
        plt.savefig(f"{output_dir}/Bar_Avg_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # -------------------------------------------------------------------------
    # 5. MATRIZ DE CORRELACIÓN
    # -------------------------------------------------------------------------
    print("Generando matriz de correlación...")
    metric_cols = [m[0] for m in qa_metrics if m[0] in df.columns]
    
    if len(metric_cols) > 1:
        plt.figure(figsize=(10, 8))
        # Seleccionamos solo las columnas numéricas relevantes
        corr = df[metric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        plt.title("Correlation Matrix of Quality Metrics")
        plt.savefig(f"{output_dir}/Correlation_Matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n¡Listo! Todos los gráficos se han guardado en: {output_dir}/")

if __name__ == "__main__":
    plot_all_metrics()