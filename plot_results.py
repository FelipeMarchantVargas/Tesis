import matplotlib
# Backend sin interfaz gráfica (ideal para scripts)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_benchmark():
    csv_path = "results/benchmark_results_final.csv"
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra el CSV en {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print("Métodos encontrados en el CSV:", df['Method'].unique())
    except Exception as e:
        print(f"Error leyendo el CSV: {e}")
        return
        
    print("--- Generando Gráficos Finales de Tesis (Incluyendo SW-SSIM) ---")
    
    # =========================================================================
    # CONFIGURACIÓN DE ESTILOS PROFESIONALES
    # =========================================================================
    methods = [
        # 1. TU MÉTODO (Proposed)
        {
            'name': 'Ours (Saliency RDO)',       
            'label': 'Proposed (Saliency RDO)',  
            'fmt': '-', 'color': '#D62728', 'marker': 'o', 'lw': 2.5, 'ms': 6
        }, 
        
        # 2. EL CONTROL (Baseline)
        {
            'name': 'Standard RDO (No Saliency)', 
            'label': 'Baseline (Standard RDO)', 
            'fmt': '--', 'color': '#1F77B4', 'marker': 'D', 'lw': 2.0, 'ms': 5
        },
        
        # 3. REFERENCIAS
        {
            'name': 'JPEG', 
            'label': 'JPEG', 
            'fmt': '-.', 'color': '#FF7F0E', 'marker': 'None', 'lw': 1.5, 'ms': 0
        },
        {
            'name': 'WebP', 
            'label': 'WebP', 
            'fmt': '-.', 'color': '#17BECF', 'marker': 'None', 'lw': 1.5, 'ms': 0
        }
    ]

    grouped = {}
    # Aseguramos que sw_ssim esté en la lista de métricas a promediar
    metrics_to_mean = ['BPP', 'lpips', 'ssim', 'ms_ssim', 'vif', 'psnr', 'sw_ssim']
    existing_metrics = [m for m in metrics_to_mean if m in df.columns]

    # Agrupar datos
    for m in methods:
        sub = df[df['Method'] == m['name']]
        if sub.empty:
            continue
        mean = sub.groupby('Param', as_index=False)[existing_metrics].mean()
        mean = mean.sort_values('BPP') 
        grouped[m['name']] = mean

    # Función de ploteo
    def create_plot(metric_col, title, ylabel, is_loss_metric=False):
        plt.figure(figsize=(8, 6))
        plot_drawn = False
        max_bpp_interest = 0.0
        
        for m in methods:
            if m['name'] not in grouped: continue
            data = grouped[m['name']]
            if metric_col not in data.columns: continue
                
            plt.plot(
                data['BPP'], data[metric_col], 
                linestyle=m['fmt'], 
                color=m['color'], 
                marker=m['marker'],
                label=m['label'], 
                linewidth=m['lw'], 
                markersize=m['ms'], 
                alpha=0.85
            )
            
            if m['name'] in ['Ours (Saliency RDO)', 'Standard RDO (No Saliency)']:
                max_bpp_interest = max(max_bpp_interest, data['BPP'].max())
            plot_drawn = True
        
        if not plot_drawn:
            plt.close()
            return

        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        
        plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.8)
        plt.minorticks_on()
        plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='#CCCCCC')
        
        # Zoom inteligente (enfocado en tu rango de operación)
        limit_x = max(1.5, max_bpp_interest * 1.2)
        plt.xlim(0, limit_x)

        os.makedirs("results/plots", exist_ok=True)
        filename = f"results/plots/RD_Curve_{metric_col}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"📈 Gráfico generado: {filename}")
        plt.close()

    # --- GENERACIÓN DE GRÁFICOS ---
    
    # 1. Métricas Globales (Standard)
    if 'psnr' in existing_metrics:
        create_plot('psnr', 'Rate-Distortion: PSNR', 'PSNR (dB) ⬆')
    if 'ssim' in existing_metrics:
        create_plot('ssim', 'Rate-Distortion: SSIM', 'SSIM ⬆')
    
    # 2. Métrica CLAVE de tu Tesis (Weighted)
    if 'sw_ssim' in existing_metrics:
        print(">>> Generando gráfico SW-SSIM (Ponderado por Saliencia)...")
        create_plot(
            'sw_ssim', 
            'Rate-Distortion: Saliency-Weighted SSIM', 
            'SW-SSIM (Ponderado por Importancia) ⬆'
        )

    # 3. Perceptual (Opcional)
    if 'lpips' in existing_metrics:
        create_plot('lpips', 'Rate-Distortion: LPIPS', 'LPIPS (Menor es mejor) ⬇', is_loss_metric=True)

if __name__ == "__main__":
    plot_benchmark()