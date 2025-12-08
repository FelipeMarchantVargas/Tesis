import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Optional, Tuple, Union
import warnings

# NOTA: Se asume que tienes la definición de la arquitectura U2NET disponible.
# Si no, deberás importarla desde tu módulo de modelos, ej:
# from src.models.u2net import U2NET
# Por ahora, usamos un Any para el tipo del modelo para evitar errores de linter estático.
ModelType = Union[nn.Module, None]

class SaliencyDetector:
    """Detector de Saliencia utilizando U2-Net o un generador Mock para pruebas.

    Esta clase encapsula la lógica de carga del modelo, preprocesamiento de la imagen,
    inferencia y postprocesamiento para obtener un mapa de probabilidad [0, 1].
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = 'auto'):
        """Inicializa el detector.

        Args:
            weights_path (Optional[str]): Ruta al archivo .pth. Si es None, se usa modo Mock.
            device (str): Dispositivo ('cpu', 'cuda', 'mps' o 'auto').
        """
        self.model: ModelType = None
        self.device = self._select_device(device)
        self.mock_mode = True

        if weights_path:
            try:
                self.model = self._load_u2net_model(weights_path)
                self.mock_mode = False
                print(f"[SaliencyDetector] Modelo cargado exitosamente en {self.device}")
            except Exception as e:
                warnings.warn(f"No se pudo cargar el modelo en {weights_path}: {e}. Usando modo Mock.")
                self.mock_mode = True
        else:
            print("[SaliencyDetector] No se proveyeron pesos. Inicializando en modo Mock.")

        # Transformaciones estándar para U2-Net (Resize 320, Normalize ImageNet)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 320), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _select_device(self, device_req: str) -> torch.device:
        """Selecciona el hardware adecuado."""
        if device_req != 'auto':
            return torch.device(device_req)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available(): # Soporte Mac M1/M2
            return torch.device('mps')
        return torch.device('cpu')

    def _load_u2net_model(self, path: str) -> nn.Module:
        """Carga la arquitectura y pesos de U2-Net."""
        # 1. Importamos la clase U2NET desde el archivo que acabamos de crear
        # Asegúrate de que src.u2net_def sea accesible
        try:
            from src.u2net_def import U2NET
        except ImportError:
            # Fallback por si ejecutas desde dentro de src
            from u2net_def import U2NET

        # 2. Instanciamos la red (3 canales entrada -> RGB, 1 salida -> Mapa)
        net = U2NET(3, 1)

        # 3. Cargamos los pesos en la CPU o GPU según corresponda
        if self.device.type == 'cpu':
            state_dict = torch.load(path, map_location='cpu')
        else:
            state_dict = torch.load(path)
            
        # 4. Asignamos los pesos al modelo
        net.load_state_dict(state_dict)
        
        # 5. Movemos el modelo al dispositivo y lo ponemos en modo evaluación
        net.to(self.device)
        net.eval()
        
        return net

    def _generate_mock_map(self, shape: Tuple[int, int]) -> np.ndarray:
        """Genera un mapa gaussiano centrado para depuración.

        Args:
            shape (Tuple[int, int]): Dimensiones (H, W) de la imagen original.

        Returns:
            np.ndarray: Mapa de saliencia simulado (H, W) en rango [0, 1].
        """
        h, w = shape
        # Crear grilla de coordenadas
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Fórmula Gaussiana: exp(-(x^2 + y^2) / 2*sigma^2)
        # Sigma controla la dispersión del centro brillante
        sigma = 0.4
        d_squared = xx**2 + yy**2
        gaussian = np.exp(-d_squared / (2.0 * sigma**2))
        
        # Asegurar rango estricto 0-1
        return gaussian.astype(np.float32)

    def get_saliency_map(self, image_rgb: np.ndarray) -> np.ndarray:
        """Obtiene el mapa de saliencia de una imagen.

        Args:
            image_rgb (np.ndarray): Imagen de entrada (H, W, 3) en uint8 o float.

        Returns:
            np.ndarray: Mapa de saliencia (H, W) float32 normalizado [0, 1].
        """
        h_orig, w_orig = image_rgb.shape[:2]

        # 1. Ruta Mock (Debugging)
        if self.mock_mode or self.model is None:
            return self._generate_mock_map((h_orig, w_orig))

        # 2. Ruta Inferencia (U2-Net)
        # Preprocesamiento
        input_tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # U2-Net retorna d1, d2, d3... d7. Nos interesa d1 (la salida final).
            d1, *_ = self.model(input_tensor)
            
            # Aplicar Sigmoid para obtener probabilidad y Normalizar
            pred = torch.sigmoid(d1[:, 0, :, :])
            
            # Post-procesamiento
            pred = pred.squeeze().cpu().numpy()

        # Resize de vuelta al tamaño original
        # Usamos interpolación lineal para mapas de probabilidad
        saliency_map = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Normalización Min-Max segura (para asegurar rango 0.0 - 1.0)
        min_val, max_val = saliency_map.min(), saliency_map.max()
        if max_val > min_val:
            saliency_map = (saliency_map - min_val) / (max_val - min_val)
        else:
            saliency_map = np.zeros_like(saliency_map)

        return saliency_map.astype(np.float32)