try: 
    from tkinter import Canvas
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import List, Tuple, Union, Optional
    import logging
except ImportError as e:
    raise ImportError(f"Faltan dependencias necesarias: {e}")

class ImagePreprocessor:
    def __init__(self,
                 target_size: Tuple[int, int] = (800, 600),
                 blur_kernel_size: Union[int, Tuple[int, int]] = (7, 7),
                 binarization_block_size: int = 13,
                 binarization_C: int = 2,
                 open_kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                 close_kernel_size: Union[int, Tuple[int, int]] = (4, 4)):
        """
        Inicializa el preprocesador de imágenes.

        Args:
            * target_size (tuple): Tamaño objetivo (ancho, alto) para estandarización.
            * blur_kernel_size (int|tuple): Tamaño del kernel para el filtro Gaussian Blur.
            * binarization_block_size (int): Tamaño del vecindario para la binarización adaptativa.
            * binarization_C (int): Constante a restar de la media/gaussiana local.
            * open_kernel_size (int|tuple): Tamaño del kernel para la operación de apertura.
            * close_kernel_size (int|tuple): Tamaño del kernel para la operación de cierre.
        """
        self.target_size = target_size
        self.blur_kernel_size = blur_kernel_size
        self.binarization_block_size = binarization_block_size
        self.binarizacion_C = binarization_C
        self.open_kernel_size = open_kernel_size
        self.close_kernel_size = close_kernel_size

        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ ImagePreprocessor inicializado correctamente")


    def process(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray_image = self._convert_to_grayscale(image)
        self.logger.debug("Imagen convertida a escala de grises.")

        # Standardize size
        gray_image = self._standardize_size(gray_image)
        self.logger.debug("Imagen redimensionada a tamaño estándar.")

        # Apply Gaussian blur
        blurred_image = self._apply_gaussian_blur(gray_image,
                                                  kernel_size=self.blur_kernel_size)
        self.logger.debug("Filtro Gaussian Blur aplicado.")

        # Apply adaptive binarization
        binarized_image = self._apply_adaptive_binarization(blurred_image,
                                                            block_size=self.binarization_block_size,
                                                            C=self.binarizacion_C)
        self.logger.debug("Binarización adaptativa aplicada.")

        # Clean binarization result
        cleaned_image = self._clean_binarization(binarized_image,
                                                 open_kernel_size=self.open_kernel_size,
                                                 close_kernel_size=self.close_kernel_size)
        self.logger.debug("Binarización limpiada con operaciones morfológicas.")
        
        return cleaned_image


    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convierte la imagen a escala de grises si es necesario.
        
        Args:
            image (np.ndarray): Imagen de entrada en formato BGR o ya en escala de grises.
        
        Returns:
            np.ndarray: Imagen en escala de grises.

        Raises:
            ValueError: Si la imagen no tiene 1 o 3 canales.
        """

        # Validar formato de imagen de entrada
        if image is None or image.size == 0:
            raise ValueError("Imagen de entrada invalida.")
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 2:
            print("Imagen ya en escala de grises.")
            return image
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray_image
    

    def _standardize_size(self, image: np.ndarray) -> np.ndarray:
        """Método interno para redimensionar manteniendo aspect ratio"""
        h, w = image.shape
        target_w, target_h = self.target_size
        
        # Calcular scaling
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Crear canvas y centrar
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    

    def _apply_gaussian_blur(self, image: np.ndarray,
                             kernel_size: Union[int, Tuple[int, int]] = (7, 7),
                             sigmaX: float = 0) -> np.ndarray:
        """
        Aplica un filtro Gaussian Blur a la imagen.

        Args:
            * image (np.ndarray): Imagen de entrada (BGR o escala de grises).
            * kernel_size (int|tuple): Tamaño del kernel (k, k) o un entero k. Debe ser impar y > 0.
            * sigmaX (float): Desviación estándar en X; 0 calcula automáticamente según kernel.

        Returns:
            * np.ndarray: Imagen filtrada con Gaussian Blur.

        Raises:
            ValueError: Si la imagen es inválida o kernel_size no es adecuado.
        """
        if image is None or image.size == 0:
            raise ValueError("Imagen de entrada inválida.")
        
        # Normalizar kernel_size a tupla (kx, ky)
        if isinstance(kernel_size, int):
            kx = ky = kernel_size
        else:
            kx, ky = kernel_size

        # Validar kernel impar y positivo
        if kx <= 0 or ky <= 0 or kx % 2 == 0 or ky % 2 == 0:
            raise ValueError("kernel_size debe ser impar y mayor que 0, p.ej. 3,5,7,...")

        # Aplicar Gaussian Blur
        blurred = cv2.GaussianBlur(image, (kx, ky), sigmaX)

        return blurred


    def _apply_adaptive_binarization(self,
                                     blurred_image: np.ndarray,
                                     max_value: int = 255,
                                     adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     threshold_type: int = cv2.THRESH_BINARY_INV,
                                     block_size: int = 11,
                                     C: int = 2) -> np.ndarray:
        """
        Aplica binarización adaptativa (adaptiveThreshold) a una imagen en escala de grises.

        Args:
            * image (np.ndarray): Imagen de entrada en escala de grises.
            * max_value (int): Valor máximo a asignar a los píxeles umbralizados (por defecto 255).
            * adaptive_method (int): Método adaptativo (cv2.ADAPTIVE_THRESH_MEAN_C o cv2.ADAPTIVE_THRESH_GAUSSIAN_C).
            * threshold_type (int): Tipo de umbral (cv2.THRESH_BINARY o cv2.THRESH_BINARY_INV).
            * block_size (int): Tamaño del vecindario (debe ser impar y >=3).
            * C (int): Constante a restar de la media/gaussiana local.

        Returns:
            np.ndarray: Imagen binarizada adaptativamente (dtype uint8).

        Raises:
            ValueError: Si la imagen no está en escala de grises o block_size no es válido.
        """
        # Validaciones
        if blurred_image is None or blurred_image.size == 0:
            raise ValueError("Imagen de entrada inválida.")
        if len(blurred_image.shape) != 2:
            raise ValueError("La imagen debe estar en escala de grises para aplicar binarización adaptativa.")
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("block_size debe ser impar y >= 3, p.ej. 3,5,7,11,...")

        # adaptiveThreshold requiere uint8 en rango [0,255]
        if blurred_image.dtype != np.uint8:
            img_uint8 = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img_uint8 = blurred_image

        # Aplicar binarización adaptativa
        adaptive = cv2.adaptiveThreshold(img_uint8,
                                         max_value,
                                         adaptive_method,
                                         threshold_type,
                                         block_size,
                                         C)
        return adaptive
    

    def _clean_binarization(self, binarized_image: np.ndarray, 
                            open_kernel_size: tuple = (2, 2),
                            close_kernel_size: tuple = (4, 4)) -> np.ndarray:
        """
        Limpia la binarizacion usando apertura y cierre morfologicos

        Args:
            binarized_image (np.ndarray): Imagen binarizada a limpiar.
            open_kernel_size (tuple): Tamaño del kernel para la apertura morfologica.
            close_kernel_size (tuple): Tamaño del kernel para el cierre morfologico.

        Returns:
            np.ndarray: Imagen binarizada limpia.
        """
        # 1. Apertura para eliminar ruido pequeno
        kernel_open = np.ones(open_kernel_size, np.uint8)
        opened_image = cv2.morphologyEx(binarized_image, cv2.MORPH_OPEN, kernel_open)

        # 2. Cierre para cerrar contornos rotos
        kernel_close = np.ones(close_kernel_size, np.uint8)
        closed_image = cv2.morphologyEx(binarized_image, cv2.MORPH_CLOSE, kernel_close) # iba opened_image

        return closed_image