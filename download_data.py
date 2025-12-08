import os
import requests
from pathlib import Path
import cv2
import numpy as np

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    return False

def setup_kodak_dataset():
    # URL oficial del mirror de Kodak Lossless
    base_url = "http://r0k.us/graphics/kodak/kodak/"
    save_dir = Path("data/kodak")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Descargando Kodak Dataset en {save_dir} ---")
    
    # Son 24 imágenes: kodim01.png hasta kodim24.png
    for i in range(1, 25):
        img_name = f"kodim{i:02d}.png"
        url = base_url + img_name
        dest = save_dir / img_name
        
        if dest.exists():
            print(f"[Saltado] {img_name} ya existe.")
            continue
            
        print(f"Descargando {img_name}...", end="")
        if download_file(url, dest):
            print(" OK")
        else:
            print(" Error")

    print("\n--- Verificación ---")
    files = list(save_dir.glob("*.png"))
    print(f"Total imágenes descargadas: {len(files)}")
    
    # Verificamos que OpenCV las pueda leer
    if len(files) > 0:
        test_img = cv2.imread(str(files[0]))
        if test_img is not None:
            print(f"Prueba de lectura exitosa: {files[0].name} - Shape: {test_img.shape}")
        else:
            print("ERROR CRÍTICO: OpenCV no pudo leer la imagen descargada.")

if __name__ == "__main__":
    setup_kodak_dataset()