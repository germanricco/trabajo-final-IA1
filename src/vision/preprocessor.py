try: 
    import cv2
    import numpy as np
    from typing import Tuple, Union
    import logging
except ImportError as e:
    raise ImportError(f"Faltan dependencias necesarias: {e}")

class ImagePreprocessor:
    def __init__(self,
                 target_size: Tuple[int, int] = (800, 600),
                 gamma: float = 1.6,
                 d_bFilter = 10,
                 binarization_block_size: int = 13,
                 binarization_C: int = 2,
                 open_kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                 close_kernel_size: Union[int, Tuple[int, int]] = (4, 4),
                 clear_border_margin: int = 20):
        """
        Inicializa el preprocesador de imágenes.

        Args:
            * target_size (tuple): Tamaño objetivo (ancho, alto) para estandarización.
            * gamma (float): Valor de gamma para la corrección gamma.
            * d_bFilter (int): Diametro del filtro bilateral.
            * binarization_block_size (int): Tamaño del vecindario para la binarización adaptativa.
            * binarization_C (int): Constante a restar de la media/gaussiana local.
            * open_kernel_size (int|tuple): Tamaño del kernel para la operación de apertura.
            * close_kernel_size (int|tuple): Tamaño del kernel para la operación de cierre.
            * clear_border_margin (int): Margen para limpiar los bordes.
        """
        self.target_size = target_size
        self.gamma = gamma
        self.d_bFilter = d_bFilter
        self.binarization_block_size = binarization_block_size
        self.binarizacion_C = binarization_C
        self.open_kernel_size = open_kernel_size
        self.close_kernel_size = close_kernel_size
        self.clear_border_margin = clear_border_margin

        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ ImagePreprocessor inicializado correctamente")


    def process(self, image: np.ndarray) -> np.ndarray:
        # Escala de Grises
        gray_image = self._convert_to_grayscale(image)
        self.logger.debug("Imagen convertida a escala de grises.")

        # Ajustar tamano
        standardized_image, padding_info = self._standardize_size(gray_image)
        self.logger.debug("Imagen redimensionada a tamaño estándar.")

        # Gamma
        contrasted_image = self._apply_gamma(standardized_image, gamma=self.gamma)
        self.logger.debug("Gamma Correction aplicado.")

        # Filtro Bilateral
        blurred_image = self._apply_bilateral_filter(contrasted_image,
                                                     self.d_bFilter)
        self.logger.debug("Filtro Bilateral aplicado (bordes preservados).")

        # Binarizacion Adaptativa
        binarized_image = self._apply_adaptive_binarization(
            blurred_image,
            block_size=self.binarization_block_size,
            C=self.binarizacion_C,
            threshold_type=cv2.THRESH_BINARY
        )
        self.logger.debug("Binarización adaptativa aplicada.")

        # Limpiar Binarizacion
        cleaned_image = self._clean_binarization(
            binarized_image,
            open_kernel_size=self.open_kernel_size,
            close_kernel_size=self.close_kernel_size
        )
        self.logger.debug("Binarización limpiada con operaciones morfológicas.")
        
        # Limpiar Bordes
        final_image = self._mask_padding_artifacts(
            cleaned_image,
            padding_info,
            safety_margin=self.clear_border_margin
        )
        self.logger.debug("Bordes limpiados.")

        # Retornar
        return final_image


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
        """
        Redimensiona la imagen manteniendo el Aspect Ratio (proporción).
        Agrega bordes negros (padding) para llegar al target_size sin deformar.
        """
        # Tomar width, height (target y original)
        target_w, target_h = self.target_size
        h, w = image.shape[:2]
        
        # Calcular factor de escala para ajustar sin recortar ni deformar
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize proporcional
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calcular padding para centrar
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        # Agregar bordes (Letterbox)
        # BORDER_CONSTANT agrega color sólido (0 = negro)
        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, 
                                       cv2.BORDER_REPLICATE)
        
        # Nos dice cuanto padding se agregó
        return new_image, (top, bottom, left, right)
    

    def _apply_gamma(self, image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """
        Aumenta el contraste en las zonas oscuras (aplasta el ruido)
        sin saturar las zonas claras.
        gamma > 1: Oscurece medios tonos (quita ruido de fondo).
        gamma < 1: Aclara sombras.
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    

    def _apply_bilateral_filter(self, image: np.ndarray, d_bFilter: int = 2) -> np.ndarray:
        """
        Reemplazo de GaussianBlur. Elimina ruido pero manteniendo los bordes.
        
        * d: Diámetro del vecindario (5-9 está bien).
        * sigmaColor: Cuánto se pueden mezclar colores (75 es estándar).
        * sigmaSpace: Qué tan lejos se mezclan píxeles (75 es estándar).
        """
        return cv2.bilateralFilter(image, d=d_bFilter, sigmaColor=100, sigmaSpace=100)


    def _apply_adaptive_binarization(self,
                                     blurred_image: np.ndarray,
                                     max_value: int = 255,
                                     adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     threshold_type: int = cv2.THRESH_BINARY,
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
            * binarized_image (np.ndarray): Imagen binarizada a limpiar.
            * open_kernel_size (tuple): Tamaño del kernel para la apertura morfologica.
            * close_kernel_size (tuple): Tamaño del kernel para el cierre morfologico.

        Returns:
            np.ndarray: Imagen binarizada limpia.
        """

        # Definir Kernels
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)

        # Closing - Soldadura
        closed_image = cv2.morphologyEx(binarized_image, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # Opening - Limpieza
        cleaned_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, open_kernel, iterations=1)
        
        return cleaned_image

    
    def _mask_padding_artifacts(self, image: np.ndarray, padding: tuple, safety_margin: int = 5) -> np.ndarray:
        """
        Pinta de negro estricto las zonas de padding, extendiéndose un poco 
        hacia adentro (safety_margin) para borrar los artefactos de borde.
        """
        top, bottom, left, right = padding
        h, w = image.shape
        
        # Pintamos de negro las barras laterales/superiores + un margen de seguridad
        # para asegurar que borramos la línea blanca del threshold.
        
        if top > 0:
            image[:top + safety_margin, :] = 0
        if bottom > 0:
            image[h - (bottom + safety_margin):, :] = 0
        if left > 0:
            image[:, :left + safety_margin] = 0
        if right > 0:
            image[:, w - (right + safety_margin):] = 0
            
        return image
    
    # PARA VISUALIZACION
    def scale_bbox_back(self, bbox_processed, original_shape):
        """
        Convierte un Bounding Box del espacio procesado (ej: 800x600 con padding)
        de vuelta al espacio de la imagen original (ej: 4000x3000).
        """
        x, y, w, h = bbox_processed
        h_orig, w_orig = original_shape[:2]
        target_w, target_h = self.target_size

        # 1. Recalcular los mismos factores que usó _standardize_size
        scale = min(target_w / w_orig, target_h / h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        # 2. Deshacer el padding (restar bordes)
        x_no_pad = x - pad_w
        y_no_pad = y - pad_h

        # 3. Deshacer el escalado (dividir por scale)
        # Usamos max(0, ...) para evitar coordenadas negativas por errores de redondeo
        x_final = max(0, int(x_no_pad / scale))
        y_final = max(0, int(y_no_pad / scale))
        w_final = int(w / scale)
        h_final = int(h / scale)

        return (x_final, y_final, w_final, h_final)
    
    def scale_mask_back(self, processed_mask: np.ndarray, original_shape: tuple) -> np.ndarray:
        """
        Devuelve la máscara al tamaño original, eliminando el padding (barras negras)
        antes de redimensionar para evitar deformaciones.
        """
        h_orig, w_orig = original_shape[:2]
        target_w, target_h = self.target_size
        
        # 1. Recalcular la escala y dimensiones que se usaron en el 'forward pass'
        scale = min(target_w / w_orig, target_h / h_orig)
        
        # Dimensiones de la imagen ÚTIL dentro del cuadro de 800x600
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        
        # 2. Calcular cuánto padding se agregó
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # 3. RECORTAR (CROP): Quitamos las barras negras
        # Tomamos solo la parte central que tiene la imagen real
        # Coordenadas: [y_inicio : y_fin, x_inicio : x_fin]
        cropped_mask = processed_mask[pad_h : pad_h + new_h, pad_w : pad_w + new_w]
        
        # 4. REDIMENSIONAR al tamaño original
        # Usamos INTER_NEAREST para mantener la máscara binaria (bordes duros, sin difuminar)
        if cropped_mask.size == 0:
            # Fallback por seguridad si el crop sale vacío
            return np.zeros((h_orig, w_orig), dtype=np.uint8)
            
        original_mask = cv2.resize(cropped_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        return original_mask