import matplotlib
matplotlib.use('Agg') # Backend sin ventana para evitar errores de GTK
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_rd_curves():
    csv_path = "results/benchmark_results_final.csv"
    if not Path(csv_path).exists():
        print("No se encuentra el CSV de resultados.")
        return

    # 1. Cargar Datos
    df = pd.read_csv(csv_path)
    
    # 2. Verificar qué nombres de métodos existen realmente en el CSV
    print("Métodos encontrados en el CSV:", df['Method'].unique())

    # 3. AGREGACIÓN
    # Promediamos los resultados de las 24 imágenes para cada punto de operación
    df_avg = df.groupby(['Method', 'Param'], as_index=False).agg({
        'BPP': 'mean',
        'PSNR': 'mean',
        'SSIM': 'mean',
        'SW-SSIM': 'mean',
        'LPIPS': 'mean'
    })

    # Ordenamos por BPP para que la línea se dibuje bien
    df_avg = df_avg.sort_values(by=['Method', 'BPP'])

    # Métricas a graficar
    metrics = [
        ("SW-SSIM", "Calidad Ponderada (Tu Fuerte)", "Mayor es mejor", "lower right"),
        ("SSIM", "Similitud Estructural", "Mayor es mejor", "lower right"),
        ("PSNR", "Fidelidad de Señal (dB)", "Mayor es mejor", "lower right"),
        ("LPIPS", "Distancia Perceptual", "Menor es mejor", "upper right")
    ]
    
    sns.set_style("whitegrid")
    
    # --- DICCIONARIOS CORREGIDOS ---
    # Usamos los nombres exactos que reportó tu error
    palette = {
        "Proposed (Saliency RDO)": "tab:red",   # Tu método = Rojo
        "Baseline (Standard RDO)": "tab:blue",  # Baseline = Azul
        "JPEG": "tab:green",
        "WebP": "tab:orange",
        # Fallbacks por si acaso (para compatibilidad con versiones viejas)
        "Ours (Multi-Mode RDO)": "tab:red",
        "Standard RDO (No Saliency)": "tab:blue"
    }
    
    dashes = {
        "Proposed (Saliency RDO)": "",          # Sólida
        "Baseline (Standard RDO)": (2,2),       # Punteada
        "JPEG": (5,5),
        "WebP": (1,1),
        "Ours (Multi-Mode RDO)": "",
        "Standard RDO (No Saliency)": (2,2)
    }
    
    markers = {
        "Proposed (Saliency RDO)": "o",
        "Baseline (Standard RDO)": "X",
        "JPEG": "s",
        "WebP": "^",
        "Ours (Multi-Mode RDO)": "o",
        "Standard RDO (No Saliency)": "X"
    }

    for metric, title, ylabel, legend_loc in metrics:
        try:
            plt.figure(figsize=(10, 6))
            
            sns.lineplot(
                data=df_avg, 
                x="BPP", 
                y=metric, 
                hue="Method", 
                style="Method",
                palette=palette,
                dashes=dashes,
                markers=markers,
                markersize=8,
                linewidth=2.5
            )
            
            plt.title(f"Curva Rate-Distortion: {title}", fontsize=14)
            plt.xlabel("Bits Per Pixel (BPP) - Promedio Dataset", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.legend(loc=legend_loc, frameon=True, shadow=True)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.minorticks_on()
            plt.xlim(0, 1.8) 
            
            # --- PARTE MODIFICADA ---
            safe_name = metric.replace("-", "_").lower()
            
            # 1. Guardar versión con auto-escala (zoom en los datos)
            output_file = f"results/rd_curve_{safe_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            # 2. Ajustar eje Y para que inicie en 0 y guardar como nueva imagen
            plt.ylim(bottom=0) # Fuerza el inicio en 0, mantiene el máximo automático
            
            # Opcional: Si es SSIM o SW-SSIM, podrías querer fijar el máximo en 1.0
            if "SSIM" in metric:
                plt.ylim(0, 1.05) 

            output_file_y0 = f"results/rd_curve_{safe_name}_y0.png"
            plt.savefig(output_file_y0, dpi=300, bbox_inches='tight')
            # -------------------------

            plt.close()
            print(f"Gráficas guardadas: {output_file} y {output_file_y0}")
            
        except Exception as e:
            print(f"Error graficando {metric}: {e}")

if __name__ == "__main__":
    plot_rd_curves()