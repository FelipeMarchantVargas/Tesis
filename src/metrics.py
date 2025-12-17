import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

class QualityMetrics:
    def __init__(self):
        # Inicializamos LPIPS (Métrica perceptual neuronal)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            self.use_lpips = True
        except:
            print("Warning: LPIPS no disponible. Se omitirá.")
            self.use_lpips = False

    def calculate_all(self, original: np.ndarray, reconstructed: np.ndarray, saliency_map: np.ndarray = None):
        """
        Calcula todas las métricas.
        Args:
            saliency_map: Mapa de prominencia [0, 1] del mismo tamaño que la imagen.
                          NECESARIO para calcular sw_ssim correctamente.
        """
        h, w = original.shape[:2]
        
        # 1. PSNR Standard
        psnr_val = psnr_func(original, reconstructed, data_range=255)
        
        # 2. SSIM Standard & Map
        # full=True nos devuelve el mapa de errores local, vital para sw_ssim
        ssim_val, ssim_map = ssim_func(
            original, reconstructed, 
            data_range=255, 
            channel_axis=2, 
            full=True
        )
        
        # 3. SW-SSIM (Saliency Weighted) - AQUÍ ESTÁ LA CLAVE
        sw_ssim_val = ssim_val # Fallback
        if saliency_map is not None:
            # El mapa SSIM viene en (H, W, 3), promediamos los canales
            ssim_map_gray = np.mean(ssim_map, axis=2)
            
            # Aseguramos dimensiones
            if saliency_map.shape != ssim_map_gray.shape:
                saliency_map = cv2.resize(saliency_map, (w, h))
            
            # Fórmula de Ponderación
            # Suma(Calidad * Importancia) / Suma(Importancia)
            numerator = np.sum(ssim_map_gray * saliency_map)
            denominator = np.sum(saliency_map)
            
            if denominator > 0:
                sw_ssim_val = numerator / denominator

        # 4. LPIPS (Perceptual)
        lpips_val = 0.0
        if self.use_lpips:
            # Convertir a Tensores [-1, 1]
            t_orig = self._to_tensor(original)
            t_rec = self._to_tensor(reconstructed)
            with torch.no_grad():
                lpips_val = self.lpips_fn(t_orig, t_rec).item()

        return {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "sw_ssim": sw_ssim_val, # Ahora sí será diferente al SSIM normal
            "lpips": lpips_val,
            "ms_ssim": 0.0, # Opcional, omitido por velocidad
            "vif": 0.0      # Opcional
        }

    def _to_tensor(self, img):
        # (H, W, C) -> (C, H, W) normalizado a [-1, 1]
        x = img.astype(np.float32) / 255.0
        x = x * 2.0 - 1.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return x