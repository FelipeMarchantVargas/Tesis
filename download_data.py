import os
import zipfile
from pathlib import Path
import urllib.request
import cv2

# Configuramos un cargador que simula un navegador real para evitar el error 403
class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"

urllib._urlopener = AppURLopener()

DATASETS_CONFIG = {
    "kodak": {
        "url_pattern": "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png",
        "range": range(1, 25),
        "type": "individual_files"
    },
    "div2k_val": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "type": "archive"
    },
    "mcmaster": {
        # Dataset de 18 imágenes de alta calidad muy usado en papers
        "url": "https://www4.comp.polyu.edu.hk/~cslzhang/code/McMaster.zip",
        "type": "archive"
    },
    "tecnick": {
        # Intentamos con el mirror directo de testimages.org
        "url": "https://testimages.org/download/sampling/TESTIMAGES.zip",
        "type": "archive"
    }
}

def download_file(url, dest_path):
    try:
        urllib._urlopener.retrieve(url, str(dest_path))
        return True
    except Exception as e:
        print(f"\n Error descargando {url}: {e}")
        return False

def extract_archive(archive_path, extract_to):
    print(f" > Extrayendo en {extract_to}...")
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f" Error de extracción: {e}")
        return False

def generate_grayscale_dataset(source_dir, target_dir):
    """Crea una versión en grises del dataset para pruebas específicas."""
    if not source_dir.exists(): return
    target_dir.mkdir(exist_ok=True)
    print(f"--- Generando Dataset Escala de Grises en {target_dir} ---")
    for img_path in source_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(target_dir / img_path.name), gray)

def setup_all():
    base_dir = Path("data")
    base_dir.mkdir(exist_ok=True)

    for name, config in DATASETS_CONFIG.items():
        target_dir = base_dir / name
        target_dir.mkdir(exist_ok=True)

        # Verificar si ya existe contenido
        if any(target_dir.iterdir()) and name != "kodak":
            print(f"[Saltado] {name.upper()} ya existe.")
            continue

        print(f"\n--- Procesando: {name.upper()} ---")

        if config["type"] == "individual_files":
            for i in config["range"]:
                url = config["url_pattern"].format(i)
                dest = target_dir / Path(url).name
                if not dest.exists():
                    print(f" Descargando {dest.name}...")
                    download_file(url, dest)
            
        elif config["type"] == "archive":
            temp_zip = target_dir / "temp.zip"
            print(f" Descargando {name} (esto puede tardar)...")
            if download_file(config["url"], temp_zip):
                extract_archive(temp_zip, target_dir)
                if temp_zip.exists(): temp_zip.unlink()
                print(f" {name.upper()} listo.")

    # Generar automáticamente el dataset de grises basado en Kodak
    generate_grayscale_dataset(base_dir / "kodak", base_dir / "kodak_gray")

if __name__ == "__main__":
    setup_all()