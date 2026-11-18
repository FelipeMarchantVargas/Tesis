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
            pred = d1[:, 0, :, :]
            
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

# ---------------------------------------------------------------------------
# Estrategias adicionales de saliencia (patrón Strategy, ver Memoria §3.2).
# El código y los pesos de TranSalNet y MobileSal viven en third_party/
# (ver setup_models.sh); no se versionan en este repositorio.
# ---------------------------------------------------------------------------
import os
import sys

THIRD_PARTY = os.environ.get(
    "SALIENCY_THIRD_PARTY",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party"),
)


def _minmax(saliency_map: np.ndarray) -> np.ndarray:
    min_val, max_val = saliency_map.min(), saliency_map.max()
    if max_val > min_val:
        return ((saliency_map - min_val) / (max_val - min_val)).astype(np.float32)
    return np.zeros_like(saliency_map, dtype=np.float32)


class U2NetSaliency(SaliencyDetector):
    """U2-Net (SOD). Alias de SaliencyDetector con la interfaz común."""
    name = "u2net"

    def __init__(self, weights_path: str = "models/u2net.pth", device: str = "auto"):
        super().__init__(weights_path=weights_path, device=device)


class TranSalNetSaliency:
    """TranSalNet (predicción de fijaciones, CNN + Transformer). variant: 'Res' o 'Dense'.

    Usa el preprocesamiento oficial: letterbox a 384x288 y recorte al tamaño original.
    """

    def __init__(self, variant: str = "Res", device: str = "auto"):
        self.name = f"transalnet_{variant.lower()}"
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
        root = os.path.join(THIRD_PARTY, "TranSalNet")
        if root not in sys.path:
            sys.path.insert(0, root)
        cwd = os.getcwd()
        os.chdir(root)  # los backbones se cargan con rutas relativas a pretrained_models/
        try:
            module = __import__(f"TranSalNet_{variant}")
            from utils.data_process import preprocess_img, postprocess_img
            model = module.TranSalNet()
            model.load_state_dict(torch.load(f"pretrained_models/TranSalNet_{variant}.pth",
                                             map_location="cpu"))
        finally:
            os.chdir(cwd)
        self._pre, self._post = preprocess_img, postprocess_img
        self.model = model.to(self.device).eval()

    def get_saliency_map(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        import tempfile
        # El preprocesamiento oficial recibe una ruta (lee con cv2 en BGR).
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            x = np.array(self._pre(tmp.name)) / 255.0
            x = torch.from_numpy(x.transpose(2, 0, 1)[None]).float().to(self.device)
            with torch.no_grad():
                pred = self.model(x).squeeze().cpu()
            pic = transforms.ToPILImage()(pred)
            smap = self._post(pic, tmp.name).astype(np.float32)
        return _minmax(smap)


class MobileSalSaliency:
    """MobileSal (SOD RGB-D eficiente) usado en modo solo RGB (depth=None)."""
    name = "mobilesal"

    def __init__(self, weights_path: Optional[str] = None, device: str = "auto", size: int = 320):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
        root = os.path.join(THIRD_PARTY, "MobileSal")
        if root not in sys.path:
            sys.path.insert(0, root)
        from models.model import MobileSal
        weights_path = weights_path or os.path.join(root, "pretrained", "mobilesal_ss.pth")
        model = MobileSal()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model = model.to(self.device).eval()
        self.size = size
        self.mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)  # BGR, como tools/test.py
        self.std = np.array([0.225, 0.224, 0.229], dtype=np.float32)

    def get_saliency_map(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
        x = (cv2.resize(bgr, (self.size, self.size)) - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1)[None]).float().to(self.device)
        with torch.no_grad():
            out = self.model(x, depth=None)
        out = out[0] if isinstance(out, (tuple, list)) else out
        smap = cv2.resize(out.squeeze().float().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        return _minmax(smap)


SALIENCY_MODELS = {
    "u2net": U2NetSaliency,
    "mobilesal": MobileSalSaliency,
    "transalnet_dense": lambda **kw: TranSalNetSaliency("Dense", **kw),
    "transalnet_res": lambda **kw: TranSalNetSaliency("Res", **kw),
}


def get_saliency_model(name: str, **kwargs):
    """Fábrica (Strategy): 'u2net', 'mobilesal', 'transalnet_dense', 'transalnet_res'."""
    return SALIENCY_MODELS[name](**kwargs)
