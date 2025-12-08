import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import lpips
import piq  # Nueva librería estrella

class QualityMetrics:
    """Calculadora de métricas de calidad de imagen (LPIPS, MS-SSIM, VIF, SSIM, PSNR)."""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[Metrics] Cargando métricas en {self.device}...")
        
        # LPIPS (Perceptual)
        self.loss_fn_lpips = lpips.LPIPS(net='vgg', verbose=False).to(self.device)
        self.loss_fn_lpips.eval()
        
        # Las otras métricas de PIQ no necesitan "cargar pesos", son funciones matemáticas directas.

    def calculate_all(self, img_true: np.ndarray, img_test: np.ndarray):
        """
        Entrada: Imágenes numpy (H, W, C) en rango [0, 255].
        Salida: Diccionario con todas las métricas.
        """
        # 1. Métricas CPU (Scikit-Image) - Estándar clásico
        # SSIM clásico (Single Scale)
        ssim_val = ssim_func(img_true, img_test, data_range=255, channel_axis=2)
        psnr_val = psnr_func(img_true, img_test, data_range=255)

        # 2. Preparar Tensores para GPU (Range [0, 1])
        t_true = self._to_tensor_01(img_true)
        t_test = self._to_tensor_01(img_test)

        # 3. Métricas GPU (PIQ + LPIPS)
        with torch.no_grad():
            # LPIPS (Requiere normalización [-1, 1])
            # Transformamos de [0,1] a [-1,1] al vuelo
            lpips_val = self.loss_fn_lpips(t_true * 2 - 1, t_test * 2 - 1).item()
            
            # MS-SSIM (Multi-Scale Structural Similarity) - Más robusta que SSIM
            # data_range=1.0 porque los tensores están en [0, 1]
            msssim_val = piq.multi_scale_ssim(t_test, t_true, data_range=1.0).item()
            
            # VIF (Visual Information Fidelity) - Excelente para degradaciones naturales
            vif_val = piq.vif_p(t_test, t_true, data_range=1.0).item()

        return {
            "psnr": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
            "ms_ssim": round(msssim_val, 4),
            "vif": round(vif_val, 4),
            "lpips": round(lpips_val, 4)
        }

    def _to_tensor_01(self, img_np: np.ndarray):
        """Convierte numpy (H,W,C) [0,255] a Tensor (1,C,H,W) [0,1]."""
        img = img_np.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) # HWC -> CHW
        return torch.from_numpy(img).unsqueeze(0).to(self.device)