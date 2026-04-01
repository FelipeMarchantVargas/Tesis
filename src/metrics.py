import numpy as np
import cv2
import torch
import torch.nn.functional as F
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
        Calcula: PSNR, SSIM, MS-SSIM, LPIPS, VIF y SW-SSIM.
        """
        h, w = original.shape[:2]
        
        # 1. PSNR Standard
        psnr_val = psnr_func(original, reconstructed, data_range=255)
        
        # 2. SSIM Standard & Map
        ssim_val, ssim_map = ssim_func(
            original, reconstructed, 
            data_range=255, 
            channel_axis=2, 
            full=True
        )
        
        # 3. SW-SSIM (Saliency Weighted)
        sw_ssim_val = ssim_val # Fallback
        if saliency_map is not None:
            ssim_map_gray = np.mean(ssim_map, axis=2)
            if saliency_map.shape != ssim_map_gray.shape:
                saliency_map = cv2.resize(saliency_map, (w, h))
            
            numerator = np.sum(ssim_map_gray * saliency_map)
            denominator = np.sum(saliency_map)
            if denominator > 0:
                sw_ssim_val = numerator / denominator

        # Preparar tensores para métricas de PyTorch (MS-SSIM y LPIPS)
        t_orig = self._to_tensor(original)
        t_rec = self._to_tensor(reconstructed)

        # 4. LPIPS (Perceptual)
        lpips_val = 0.0
        if self.use_lpips:
            with torch.no_grad():
                lpips_val = self.lpips_fn(t_orig, t_rec).item()

        # 5. MS-SSIM (Multi-Scale SSIM) - Implementación PyTorch interna
        ms_ssim_val = self._calculate_msssim_torch(t_orig, t_rec)

        # 6. VIF (Visual Information Fidelity) - Aproximación simple en Pixel Domain
        vif_val = self._calculate_vif_pixel_domain(original, reconstructed)

        return {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "sw_ssim": sw_ssim_val,
            "ms_ssim": ms_ssim_val,
            "lpips": lpips_val,
            "vif": vif_val
        }

    def _to_tensor(self, img):
        # (H, W, C) -> (1, C, H, W) normalizado a [-1, 1] para LPIPS, [0, 1] para MS-SSIM
        x = img.astype(np.float32) / 255.0
        x = x * 2.0 - 1.0 # Para LPIPS [-1, 1]
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return x

    def _calculate_msssim_torch(self, img1, img2, method='product'):
        """ Implementación simplificada de MS-SSIM en PyTorch """
        # Ajustar rango de [-1, 1] a [0, 1] para SSIM calc
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(self.device)
        levels = weights.shape[0]
        mssim = []
        mcs = []
        
        for _ in range(levels):
            sim, cs = self._ssim_torch(img1, img2, size_average=True)
            mssim.append(sim)
            mcs.append(cs)
            # Downsample
            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)
        
        # MS-SSIM = Prod(MCS_i^weight_i) * SSIM_last^weight_last
        # Usamos clamp para evitar log(0)
        p1 = mcs[0:levels-1] ** weights[0:levels-1]
        p2 = mssim[levels-1] ** weights[levels-1]
        
        ms_ssim_val = torch.prod(p1) * p2
        return ms_ssim_val.item()

    def _ssim_torch(self, img1, img2, size_average=True):
        # Función auxiliar básica de SSIM/CS para tensores
        mean1 = F.avg_pool2d(img1, 11, 1, 5) # window size 11
        mean2 = F.avg_pool2d(img2, 11, 1, 5)
        sigma1 = F.avg_pool2d(img1**2, 11, 1, 5) - mean1**2
        sigma2 = F.avg_pool2d(img2**2, 11, 1, 5) - mean2**2
        sigma12 = F.avg_pool2d(img1*img2, 11, 1, 5) - mean1*mean2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1**2 + sigma2**2 + C2
        
        cs = torch.mean(v1 / v2)  # Contrast structure
        ssim = torch.mean(((2 * mean1 * mean2 + C1) * v1) / ((mean1**2 + mean2**2 + C1) * v2))
        return ssim, cs

    def _calculate_vif_pixel_domain(self, ref, dist):
        # VIF pixel-based simple approximation
        sigma_nsq = 2.0
        ref = ref.astype(np.float32)
        dist = dist.astype(np.float32)
        
        mu1 = cv2.GaussianBlur(ref, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(dist, (11, 11), 1.5)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(ref * ref, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(dist * dist, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(ref * dist, (11, 11), 1.5) - mu1_mu2
        
        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12
        
        g = np.where(sigma1_sq < 1e-10, 0, g)
        sv_sq = np.where(sigma1_sq < 1e-10, sigma2_sq, sv_sq)
        sv_sq = np.where(sv_sq < 1e-10, 1e-10, sv_sq)
        
        num = np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq))
        den = np.log10(1 + sigma1_sq / sigma_nsq)
        
        vif_val = np.sum(num) / np.sum(den)
        return np.clip(vif_val, 0, 1)