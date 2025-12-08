import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_benchmark():
    csv_path = "results/benchmark_results_full.csv" # Asegúrate de que este nombre sea correcto
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra el CSV en {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error leyendo el CSV: {e}")
        return
        
    print("--- Generando 4 Gráficos Comparativos ---")
    
    methods = [
        {'name': 'Ours', 'label': 'CNN-Quadtree (Propuesto)', 'fmt': 'g-s', 'lw': 2},
        {'name': 'JPEG', 'label': 'JPEG (Standard)', 'fmt': 'r--o', 'lw': 2},
        {'name': 'WebP', 'label': 'WebP (Modern)', 'fmt': 'b-.^', 'lw': 2}
    ]

    # Agrupamos los datos
    grouped = {}
    # Columnas numéricas que queremos promediar
    metrics_to_mean = ['BPP', 'lpips', 'ssim', 'ms_ssim', 'vif']

    for m in methods:
        sub = df[df['Method'] == m['name']]
        if sub.empty:
            print(f"Advertencia: No hay datos para el método '{m['name']}'. Saltando.")
            continue
            
        # --- CORRECCIÓN AQUÍ ---
        # Seleccionamos explícitamente las columnas numéricas para promediar
        # Agrupamos por 'Param' y calculamos la media solo de las métricas
        mean = sub.groupby('Param', as_index=False)[metrics_to_mean].mean()
        
        # Ordenamos por BPP para que la línea del gráfico se dibuje correctamente
        mean = mean.sort_values('BPP')
        grouped[m['name']] = mean

    # Función helper para plotear
    def create_plot(metric_col, title, ylabel, ascending_better=True):
        plt.figure(figsize=(10, 6))
        plot_drawn = False
        
        for m in methods:
            if m['name'] not in grouped:
                continue
            data = grouped[m['name']]
            # Verificar que la métrica existe en los datos
            if metric_col not in data.columns:
                print(f"Advertencia: Métrica '{metric_col}' no encontrada para '{m['name']}'.")
                continue
                
            plt.plot(data['BPP'], data[metric_col], m['fmt'], label=m['label'], linewidth=m['lw'])
            plot_drawn = True
        
        if not plot_drawn:
            print(f"No se pudo generar el gráfico para {metric_col} (faltan datos).")
            plt.close()
            return

        plt.title(f'{title} vs Tasa de Bits', fontsize=14)
        plt.xlabel('Bits Por Píxel (BPP)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11)
        
        # Guardar
        filename = f"results/curve_{metric_col}.png"
        # Asegurar que el directorio results existe
        os.makedirs("results", exist_ok=True)
        plt.savefig(filename, dpi=300)
        print(f"Gráfico guardado: {filename}")
        plt.close()

    # --- Generar los 4 Gráficos ---
    # Verificamos que las columnas existan en el DF original antes de intentar plotear
    available_metrics = df.columns.tolist()
    
    if 'lpips' in available_metrics:
        create_plot('lpips', 'Calidad Perceptual (LPIPS)', 'LPIPS [Menor es mejor]', ascending_better=True)
    if 'ssim' in available_metrics:
        create_plot('ssim', 'Similitud Estructural (SSIM)', 'SSIM [Mayor es mejor]', ascending_better=False)
    if 'ms_ssim' in available_metrics:
        create_plot('ms_ssim', 'Similitud Multiescala (MS-SSIM)', 'MS-SSIM [Mayor es mejor]', ascending_better=False)
    if 'vif' in available_metrics:
        create_plot('vif', 'Fidelidad de Info. Visual (VIF)', 'VIF [Mayor es mejor]', ascending_better=False)

if __name__ == "__main__":
    plot_benchmark()