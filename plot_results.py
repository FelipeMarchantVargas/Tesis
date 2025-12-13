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
    except Exception as e:
        print(f"Error leyendo el CSV: {e}")
        return
        
    print("--- Generando Gráficos Finales de Tesis (Estilo Académico) ---")
    
    # =========================================================================
    # CONFIGURACIÓN DE ESTILOS PROFESIONALES
    # =========================================================================
    methods = [
        # 1. TU MÉTODO (El Protagonista)
        # Rojo sólido (#D62728), Marcador Círculo, Línea gruesa
        {
            'name': 'Ours (optimized)',         # Nombre en el CSV
            'label': 'Proposed (Saliency-Guided)', # Nombre en el Gráfico (Formal)
            'fmt': '-', 'color': '#D62728', 'marker': 'o', 'lw': 2.5, 'ms': 6
        }, 
        
        # 2. EL CONTROL CIENTÍFICO (El Rival Directo)
        # Azul (#1F77B4), Guiones, Marcador Diamante
        {
            'name': 'Standard QT (Interp)', 
            'label': 'Control (Variance + Interp.)', 
            'fmt': '--', 'color': '#1F77B4', 'marker': 'D', 'lw': 1.8, 'ms': 5
        },
        
        # 3. EL BASELINE (El Clásico)
        # Negro o Gris Oscuro (#333333), Guiones, Marcador X
        {
            'name': 'Standard QT', 
            'label': 'Baseline (Standard QT)', 
            'fmt': '--', 'color': '#333333', 'marker': 'x', 'lw': 1.5, 'ms': 5
        }, 
        
        # 4. LA ABLACIÓN (Prueba interna)
        # Magenta (#9467BD), Punteado, Marcador Triángulo
        {
            'name': 'Ours (Ablation-Blocks)', 
            'label': 'Ablation (Saliency + Blocks)', 
            'fmt': ':', 'color': '#9467BD', 'marker': '^', 'lw': 1.5, 'ms': 5
        }, 
        
        # 5. ESTÁNDARES (Contexto)
        # Naranja y Cian, Líneas dash-dot
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

    # Función de ploteo mejorada
    def create_plot(metric_col, title, ylabel, is_loss_metric=False):
        # Tamaño académico estándar (bueno para papers/PDFs)
        plt.figure(figsize=(8, 6))
        plot_drawn = False
        
        max_bpp_interest = 0.0
        
        for m in methods:
            if m['name'] not in grouped: continue
            data = grouped[m['name']]
            if metric_col not in data.columns: continue
                
            # Plot manual para mayor control de estilos
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
            
            # Rastreo para zoom
            if m['name'] in ['JPEG', 'WebP', 'Ours (optimized)']:
                max_bpp_interest = max(max_bpp_interest, data['BPP'].max())
            
            plot_drawn = True
        
        if not plot_drawn:
            plt.close()
            return

        # --- ESTÉTICA ACADÉMICA ---
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        
        # Grid suave
        plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.8)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
        plt.minorticks_on()
        
        # Leyenda limpia
        plt.legend(fontsize=10, loc='best', frameon=True, framealpha=0.9, edgecolor='#CCCCCC')
        
        # Zoom inteligente (Limitamos el eje X para ver los detalles importantes)
        # Si JPEG llega hasta 1.5, mostramos hasta 1.8. Si tu método llega a 1.2, mostramos hasta 1.5.
        limit_x = max(1.5, max_bpp_interest * 1.1)
        # Tope máximo razonable (más allá de 3 BPP no es compresión útil)
        limit_x = min(limit_x, 3.5) 
        plt.xlim(0, limit_x)

        # Guardar
        os.makedirs("results/plots", exist_ok=True)
        filename = f"results/plots/RD_Curve_{metric_col}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300) # Alta resolución
        print(f"📈 Gráfico Académico generado: {filename}")
        plt.close()

    # Generar gráficos
    if 'psnr' in existing_metrics:
        create_plot('psnr', 'Relación Señal-Ruido (PSNR)', 'PSNR (dB) [Mayor es mejor] ⬆')
    if 'ssim' in existing_metrics:
        create_plot('ssim', 'Similitud Estructural (SSIM)', 'SSIM [Mayor es mejor] ⬆')
    if 'lpips' in existing_metrics:
        create_plot('lpips', 'Calidad Perceptual (LPIPS)', 'LPIPS [Menor es mejor] ⬇', is_loss_metric=True)
    if 'ms_ssim' in existing_metrics:
        create_plot('ms_ssim', 'Similitud Multiescala (MS-SSIM)', 'MS-SSIM [Mayor es mejor] ⬆')
    if 'sw_ssim' in existing_metrics:
        create_plot('sw_ssim', 'SSIM Ponderado por Saliencia (SW-SSIM)', 'SW-SSIM [Mayor es mejor] ⬆')

if __name__ == "__main__":
    plot_benchmark()