import torch
import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from piq import multi_scale_ssim, vif_p

class QualityMetrics:
    """
    Motor de cálculo de métricas de calidad de imagen.
    Incluye métricas estándar (PSNR, SSIM), perceptuales (LPIPS, VIF)
    y ponderadas por atención (SW-SSIM).
    """
    
    def __init__(self, use_gpu: bool = True):
        # Configurar dispositivo (CUDA si es posible para LPIPS/MS-SSIM)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"[QualityMetrics] Iniciando motor de métricas en: {self.device}")
        
        # Cargar modelo LPIPS (AlexNet es el estándar para evaluación perceptual)
        # Se descarga la primera vez que se ejecuta
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.loss_fn.eval()

    def calculate_all(self, img_true: np.ndarray, img_test: np.ndarray, saliency_map: np.ndarray = None) -> dict:
        """
        Calcula todas las métricas comparando la imagen original con la reconstruida.

        Args:
            img_true (np.ndarray): Imagen original RGB (H, W, 3) uint8 [0-255].
            img_test (np.ndarray): Imagen reconstruida RGB (H, W, 3) uint8 [0-255].
            saliency_map (np.ndarray, optional): Mapa de saliencia (H, W) float [0-1].
                                                 Requerido para calcular SW-SSIM.

        Returns:
            dict: Diccionario con las claves 'psnr', 'ssim', 'sw_ssim', 'lpips', 'ms_ssim', 'vif'.
        """
        results = {}

        # ---------------------------------------------------------------------
        # 1. MÉTRICAS CPU (Scikit-Image) - Estándares de Ingeniería
        # ---------------------------------------------------------------------
        
        # A. PSNR
        try:
            # data_range=255 es crítico para imágenes uint8
            results['psnr'] = psnr(img_true, img_test, data_range=255)
        except Exception:
            results['psnr'] = 0.0

        # B. SSIM Estándar y Mapa de SSIM
        # win_size=11 es el default del paper original de Wang et al.
        # channel_axis=2 indica que es RGB (H,W,C)
        try:
            score_ssim, ssim_full_map = ssim(
                img_true, img_test, 
                win_size=11, 
                channel_axis=2, 
                data_range=255, 
                full=True # Necesario para obtener el mapa pixel a pixel
            )
            results['ssim'] = score_ssim
        except Exception:
            score_ssim = 0.0
            ssim_full_map = None
            results['ssim'] = 0.0

        # C. SW-SSIM (Saliency Weighted SSIM) - LA CLAVE DE TU TESIS
        # Mide la calidad estructural ponderada por la importancia visual.
        if saliency_map is not None and ssim_full_map is not None:
            # ssim_full_map es (H, W, 3), promediamos los canales para tener (H, W)
            ssim_map_gray = np.mean(ssim_full_map, axis=2)
            
            # Verificación de dimensiones por seguridad
            if ssim_map_gray.shape == saliency_map.shape:
                # Fórmula: Suma(Error * Importancia) / Suma(Importancia)
                weighted_sum = np.sum(ssim_map_gray * saliency_map)
                total_weight = np.sum(saliency_map) + 1e-8 # Epsilon para evitar div/0
                results['sw_ssim'] = weighted_sum / total_weight
            else:
                # Si las dimensiones no calzan (raro), fallback al SSIM normal
                results['sw_ssim'] = score_ssim
        else:
            # Si no hay mapa de saliencia, SW-SSIM es igual a SSIM
            results['sw_ssim'] = score_ssim

        # ---------------------------------------------------------------------
        # 2. MÉTRICAS GPU (PyTorch/PIQ/LPIPS) - Perceptuales Modernas
        # ---------------------------------------------------------------------
        
        # Preparación de Tensores
        # Rango [0, 1] para PIQ (MS-SSIM, VIF)
        t_true_01 = self._to_tensor_01(img_true)
        t_test_01 = self._to_tensor_01(img_test)
        
        # Rango [-1, 1] para LPIPS (Requerimiento específico de la librería)
        t_true_norm = t_true_01 * 2.0 - 1.0
        t_test_norm = t_test_01 * 2.0 - 1.0

        with torch.no_grad():
            # D. LPIPS (Menor es mejor)
            try:
                results['lpips'] = self.loss_fn(t_true_norm, t_test_norm).item()
            except Exception:
                results['lpips'] = 1.0 # Valor alto (malo) por defecto

            # E. MS-SSIM (Multi-Scale SSIM)
            try:
                # data_range=1.0 porque los tensores están normalizados
                results['ms_ssim'] = multi_scale_ssim(t_true_01, t_test_01, data_range=1.0).item()
            except Exception:
                results['ms_ssim'] = 0.0 # Falla en imágenes muy pequeñas (<160px)

            # F. VIF (Visual Information Fidelity)
            try:
                results['vif'] = vif_p(t_true_01, t_test_01, data_range=1.0).item()
            except Exception:
                results['vif'] = 0.0

        return results

    def _to_tensor_01(self, img: np.ndarray) -> torch.Tensor:
        """
        Convierte numpy HWC [0,255] -> Torch Tensor CHW [0,1] en GPU/CPU.
        """
        # Normalizar a float 0-1
        x = img.astype(np.float32) / 255.0
        # Transponer de (H, W, C) a (C, H, W)
        x = np.transpose(x, (2, 0, 1))
        # Convertir a tensor y agregar dimensión de batch (1, C, H, W)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)