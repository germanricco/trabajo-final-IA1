import cv2
import numpy as np
from typing import List, Dict, Tuple
import math

class FeatureExtractor:
    def __init__(self):
        """
        Inicializa el extractor de caracteristicas con parametros configurables
        """
        self._epsilon_factor = 0.02  # Para aproximación poligonal
        self._min_contour_length = 50  # Longitud mínima de contorno válido

        self.feature_names = [
            'circularity',
            'aspect_ratio', 
            'solidity',
            'perimeter_area_ratio',
            'hu_moment_1',
            'hu_moment_2'
        ]

        print("✅ FeatureExtractor inicializado correctamente")

    def extract_features(self,
                         bounding_boxes: List[Tuple],
                         masks: List[np.ndarray]) -> List[Dict]:

        """
        Extrae caracteristicas para cada objeto detectado en la imagen

        Args:
            bounding_boxes (List[Tuple]): Lista de cajas delimitadoras
            masks (List[np.ndarray]): Lista de mascaras

        Returns:
            List[Dict]: Lista de diccionarios con las caracteristicas
        """

        features_list = []

        print(f"🚀 Iniciando extracción de características para {len(bounding_boxes)} objetos")

        for i, (bbox, mask) in enumerate(zip(bounding_boxes, masks)):
            print(f"   🔍 Procesando objeto {i+1}/{len(bounding_boxes)}...")
            
            try:
                features = self._extract_single_object_features(bbox, mask, i+1)
                features_list.append(features)
                print(f"   ✅ Objeto {i+1}: {len(features)} características extraídas")

            except Exception as e:
                print(f"   ❌ Error procesando objeto {i+1}: {e}")
                features_list.append(self._create_default_features(i+1, bbox))
        
        print(f"Extraccion completada: {len(features_list)} objetos procesados")
        return features_list

    def _extract_single_object_features(self,
                                      bbox: Tuple, 
                                      mask: np.ndarray,
                                      obj_id: int) -> Dict:
        """
        Extrae características para un solo objeto

        Args:
            bbox (Tuple): Caja delimitadora del objeto
            mask (np.ndarray): Mascara binaria del objeto
            obj_id (int): Identificador unico del objeto

        Returns:
            Dict: Diccionario con las caracteristicas
        """

        # Validar entrada
        if mask is None or mask.size == 0:
            print(f"   ⚠️ Objeto {obj_id}: Máscara vacía, usando características por defecto")
            return self._create_default_features(obj_id, bbox)
        
        # Obtener contorno principal
        contour = self._get_contour_from_mask(mask)
        if contour is None:
            print(f"   ⚠️ Objeto {obj_id}: No se pudo extraer contorno, usando características por defecto")
            return self._create_default_features(obj_id, bbox)
        
        # Validar contorno
        if len(contour) < 3:
            print(f"   ⚠️ Objeto {obj_id}: Contorno muy pequeño, usando características por defecto")
            return self._create_default_features(obj_id, bbox)
        
        # 1. Calcular propiedades basicas
        basic_props = self._calculate_basic_properties(contour, mask)

        # Validar propiedades basicas
        if basic_props["area"] == 0:
            print(f"   ⚠️ Objeto {obj_id}: Área insuficiente, usando características por defecto")
            return self._create_default_features(obj_id, bbox)
        
        # 2. Calcular propiedades avanzadas
        advanced_props = self._calculate_advanced_properties(contour, basic_props, bbox)

        features = {
            "object_id": obj_id,
            "bounding_box": bbox,
            **advanced_props
        }
        
        return features

    def _calculate_basic_properties(self, contour: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Calcula propiedades basicas del contorno y mascara

        Args:
            contour (np.ndarray): Contorno del objeto
            mask (np.ndarray): Mascara binaria del objeto

        Returns:
            Dict: Diccionario con las propiedades basicas calculadas
        """

        # Calcular propiedades geometricas fundamentales
        area = float(np.sum(mask > 0))
        perimeter = float(cv2.arcLength(contour, True))
        
        # Calcular Convex hull
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        
        # Calcular momentos de Hu
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)

        # Normalizar
        hu1 = self._normalize_hu_moment(hu_moments[0][0])
        hu2 = self._normalize_hu_moment(hu_moments[1][0])
        
        return {
            'area': area,
            'perimeter': perimeter,
            'hull_area': hull_area,
            'hu1': float(hu1),
            'hu2': float(hu2),
            'hull': hull
        }    


    def _calculate_advanced_properties(self, contour: np.ndarray,
                                    basic_props: Dict,
                                    bbox: Tuple) -> Dict:
        """
        Calcula propiedades avanzadas usando propiedades básicas precomputadas.

        Args:
            contour (np.ndarray): Contorno del objeto
            basic_props (Dict): Diccionario con las propiedades basicas calculadas
            bbox (Tuple): Caja delimitadora del objeto

        Returns:
            Dict: Diccionario con caracteristicas avanzadas para clasificacion
        """

        # Tomo todos las propiedades basicas
        area = basic_props['area']
        perimeter = basic_props['perimeter']
        hull_area = basic_props['hull_area']
        hu1 = basic_props['hu1']
        hu2 = basic_props['hu2']

        # 1. Circularidad - Medida de que tan circular es la forma
        circularity = self._compute_circularity(area, perimeter)

        # 2. Aspect Ratio - Relacion de aspecto usando rectangulo rotado minimo
        aspect_ratio = self._compute_oriented_aspect_ratio(contour)
            
        # 3. Solidez - Relacion entre el area y area del convex hull
        solidity = self._compute_solidity(area, hull_area)
        
        # 4. Relacion perimetro-area - Modela la complejidad del contorno
        perimeter_area_ratio = self._compute_perimeter_area_ratio(perimeter, area)  # Evitar división por cero
        
        # 5. HU MOMENTS (ya calculados y normalizados) - Invariantes de forma
        
        return {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'perimeter_area_ratio': perimeter_area_ratio,
            'hu_moment_1': hu1,
            'hu_moment_2': hu2
        }
    
    def _compute_circularity(self, area: float, perimeter: float) -> float:
        """Calcula la circularidad: 4π * área / perímetro²"""
        if perimeter > 0:
            return (4 * np.pi * area) / (perimeter ** 2)
        return 0.0
    
    def _compute_oriented_aspect_ratio(self, contour: np.ndarray) -> float:
        """Calcula el aspect ratio usando el rectángulo rotado mínimo"""
        if len(contour) < 5:  # Se necesitan al menos 5 puntos para un rectángulo rotado
            return 0.0
            
        try:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            if min(width, height) > 0:
                return max(width, height) / min(width, height)
        except Exception as e:
            print(f"   ⚠️ Error calculando aspect ratio: {e}")
            
        return 0.0
    
    def _compute_solidity(self, area: float, hull_area: float) -> float:
        """Calcula la solidez: área / área_del_convex_hull"""
        if hull_area > 0:
            return area / hull_area
        return 0.0
    
    def _compute_perimeter_area_ratio(self, perimeter: float, area: float) -> float:
        """Calcula la relación perímetro-área"""
        if area > 0:
            return perimeter / area
        return 0.0
    
    def _normalize_hu_moment(self, hu_value: float) -> float:
        """
        Normaliza un momento de Hu usando transformación logarítmica.
        
        Referencia: 
        https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944
        """
        if hu_value == 0:
            return 0.0
        
        # Transformación logarítmica para mejorar estabilidad numérica
        normalized = -np.sign(hu_value) * np.log10(np.abs(hu_value) + 1e-10)
        return float(normalized)


    def _get_contour_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extrae el contorno principal de una máscara binaria.
        """
        try:
            # Encontrar contornos
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # Devolver el contorno más grande por área
            return max(contours, key=cv2.contourArea)
            
        except Exception as e:
            print(f"   ⚠️ Error extrayendo contorno: {e}")
            return None
    
    
    def _create_default_features(self, obj_id: int, bbox: Tuple) -> Dict:
        """
        Crea características por defecto para objetos problemáticos
        """
        return {
            "object_id": obj_id,
            "bounding_box": bbox,
            'circularity': 0.0,
            'aspect_ratio': 0.0,
            'solidity': 0.0,
            'perimeter_area_ratio': 0.0,
            'hu_moment_1': 0.0,
            'hu_moment_2': 0.0
        }
    
    def get_feature_vector(self, features_dict: Dict) -> np.ndarray:
        """
        Convierte un diccionario de características a vector numérico.

        Args:
            features_dict: Diccionario de características

        Returns:
            Vector numérico con las características
        """
        return np.array([
            features_dict['circularity'],
            features_dict['aspect_ratio'],
            features_dict['solidity'],
            features_dict['perimeter_area_ratio'],
            features_dict['hu_moment_1'],
            features_dict['hu_moment_2']
        ])