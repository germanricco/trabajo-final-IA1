import cv2
import numpy as np
from typing import List, Dict, Tuple
import math
from src.agent.ContourManager import ContourManager
import logging

class FeatureExtractor:
    def __init__(self):
        """
        Inicializa el extractor de caracteristicas con parametros configurables
        """
        self._epsilon_factor = 0.02     # Para aproximación poligonal
        self._min_contour_length = 50   # Longitud mínima de contorno válido

        self.feature_names = [
            'circularity',
            'aspect_ratio', 
            'solidity',
            'perimeter_area_ratio',
            'hu_moment_1',
            'hu_moment_2'
        ]

        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ FeatureExtractor inicializado correctamente")

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
        self.logger.debug("Iniciando extracción de características para {len(bounding_boxes)} objetos")

        for i, (bbox, mask) in enumerate(zip(bounding_boxes, masks)):
            self.logger.debug("Procesando objeto {i+1}/{len(bounding_boxes)}...")
            
            try:
                features = self._extract_single_object_features(bbox, mask, i+1)
                features_list.append(features)
                self.logger.debug("Objeto {i+1}: {len(features)} características extraídas")

            except Exception as e:
                self.logger.error("Error extrayendo caracteristicas de objeto {i+1}: {e}")
                features_list.append(self._create_default_features(i+1, bbox))
        
        self.logger.debug("Extraccion completada: {len(features_list)} objetos procesados")
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
            self.logger.warning("Objeto {obj_id}: Máscara vacía, usando características por defecto")
            return self._create_default_features(obj_id, bbox)
        
        # Inicializar el gestor de contornos
        contour_manager = ContourManager(mask)
        
        # Calcular propiedades basicas
        basic_props = self._calculate_basic_properties(contour_manager, mask)
        self.logger.debug("Caracteristicas basicas calculadas")

        # Calcular propiedades avanzadas
        advanced_props = self._calculate_advanced_properties(contour_manager, basic_props)
        self.logger.debug("Caracteristicas avanzadas calculadas")

        # Ordenar y retornar
        features = {
            "object_id": obj_id,
            **basic_props,
            **advanced_props
        }
        return features

    def _calculate_basic_properties(self,
                                    contour_manager: ContourManager,
                                    mask: np.ndarray) -> Dict:
        """
        Calcula propiedades basicas del contorno y mascara

        Args:
            contour (np.ndarray): Contorno del objeto
            mask (np.ndarray): Mascara binaria del objeto

        Returns:
            Dict: Diccionario con las propiedades basicas calculadas
        """
        # Area externa
        external_area = cv2.contourArea(contour_manager.external_contour)

        # Area neta
        actual_area = contour_manager.get_actual_area()

        # Perímetro del contorno externo
        external_perimeter = cv2.arcLength(contour_manager.external_contour, True)
        
        # Area del casco convexo
        hull_area = float(cv2.contourArea(contour_manager.convex_hull))
        
        # Calcular momentos de Hu
        moments = cv2.moments(contour_manager.external_contour)
        hu_moments = cv2.HuMoments(moments)

        # Normalizar
        hu1 = self._normalize_hu_moment(hu_moments[0][0])
        hu2 = self._normalize_hu_moment(hu_moments[1][0])
        
        return {
            'external_area': external_area,     # Para circularidad
            'area': actual_area,                # Area neta para solidez
            'perimeter': external_perimeter,    # Para circularidad y compacidad
            'hull_area': hull_area,
            'hu1': float(hu1),
            'hu2': float(hu2),
        }    


    def _calculate_advanced_properties(self, contour_manager: ContourManager,
                                    basic_props: Dict) -> Dict:
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
        external_area = basic_props['external_area']
        area = basic_props['area']
        perimeter = basic_props['perimeter']
        hull_area = basic_props['hull_area']

        # Solidez
        solidity = area / hull_area if hull_area > 0 else 0

        # Circularidad - Medida de que tan circular es la forma
        circularity = (4 * np.pi * external_area) / (perimeter ** 2) if perimeter > 0 else 0

        # Compacidad considerando agujeros
        compactness = (perimeter ** 2) / external_area if external_area > 0 else 0

        # Aspect Ratio - Relacion de aspecto usando rectangulo rotado minimo
        min_rect = contour_manager.min_area_rectangle
        width, height = min_rect[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
        
        # Clasificacion de Agujeros (reemplaza logica booleana de 'has_hole')
        hole_classification = self._classify_hole(contour_manager, basic_props)

        # Compactness: Factor de Forma. 
        return {
            'solidity': solidity,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            "compactness": compactness,     

            "has_structural_hole": hole_classification["has_structural_hole"],
            "hole_confidence": hole_classification["hole_confidence"]
        }
    
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
    

    def _classify_hole(self, contour_manager: ContourManager, basic_props: Dict) -> Dict:
        """
        Evalúa si el objeto tiene un agujero estructural real utilizando un sistema de puntuación.
        Devuelve un diccionario con 'has_structural_hole' (booleano) y 'hole_confidence' (float).
        """

        # Si no hay contornos internos, definitivamente no hay agujero.
        if not contour_manager.has_hole():
            return {"has_structural_hole": False, "hole_confidence": 0.0}

        # 1. Relación de Área del Agujero (Hole Area Ratio)
        external_area = cv2.contourArea(contour_manager.external_contour)
        hole_area = cv2.contourArea(contour_manager.internal_contour)
        hole_area_ratio = hole_area / external_area if external_area > 0 else 0

        # 2. Circularidad del Agujero
        hole_perimeter = cv2.arcLength(contour_manager.internal_contour, True)
        hole_circularity = (4 * np.pi * hole_area) / (hole_perimeter ** 2) if hole_perimeter > 0 else 0

        # 3. Centrado del Agujero
        # - Encuentra el centro del contorno externo y del interno
        M_ext = cv2.moments(contour_manager.external_contour)
        M_int = cv2.moments(contour_manager.internal_contour)
        
        if M_ext["m00"] != 0 and M_int["m00"] != 0:
            cX_ext = int(M_ext["m10"] / M_ext["m00"])
            cY_ext = int(M_ext["m01"] / M_ext["m00"])
            cX_int = int(M_int["m10"] / M_int["m00"])
            cY_int = int(M_int["m01"] / M_int["m00"])
            
            # Calcular la distancia entre centros
            centroid_distance = np.sqrt((cX_ext - cX_int)**2 + (cY_ext - cY_int)**2)
            # Normalizar la distancia por el diámetro aproximado del objeto externo
            equivalent_diameter = np.sqrt(4 * external_area / np.pi)
            normalized_centroid_distance = centroid_distance / equivalent_diameter
        else:
            normalized_centroid_distance = 1.0  # Valor por defecto (descentrado)

        # Sistema de Puntuación
        score = 0.0

        # A. Puntuación por Tamaño del Agujero
        if hole_area_ratio > 0.15:  # Agujeros muy grandes puntúan alto
            score += 2.0
        elif hole_area_ratio > 0.05:  # Agujeros de tamaño medio
            score += 1.0

        # B. Puntuación por Circularidad
        if hole_circularity > 0.75:  # Agujeros muy circulares
            score += 2.0
        elif hole_circularity > 0.5:
            score += 1.0

        # C. Puntuación por Centrado
        if normalized_centroid_distance < 0.1:  # Muy bien centrado
            score += 2.0
        elif normalized_centroid_distance < 0.2:  # Aceptablemente centrado
            score += 1.0

        # Umbral de decisión (ajustable mediante validación)
        HOLE_CONFIDENCE_THRESHOLD = 2.8
        has_structural_hole = score >= HOLE_CONFIDENCE_THRESHOLD
        hole_confidence = min(score / HOLE_CONFIDENCE_THRESHOLD, 1.0)  # Normalizado a 0-1

        return {
            "has_structural_hole": has_structural_hole,
            "hole_confidence": hole_confidence
        }