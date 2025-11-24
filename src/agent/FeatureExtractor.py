
from src.agent.ContourManager import ContourManager
from src.agent.ContourManager import GeometricProperties
from typing import Dict, List, Tuple, Any
import numpy as np
import logging

class FeatureExtractor:
    """
    Extrae características optimizadas para clasificación con K-Means.
    Calcula cada característica una sola vez y las reutiliza.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        #! Borrar si no se usan
        self._feature_definitions = {
            # Formato: 'nombre_feature': ('descripción', función_extracción)
            'area': ('Área total del objeto', lambda p: p.area),
            'perimeter': ('Longitud del contorno externo', lambda p: p.perimeter),
            'solidity': ('Densidad del objeto (área/hull_area)', lambda p: p.solidity),
            'circularity': ('Grado de circularidad', lambda p: p.circularity),
            'aspect_ratio': ('Relación de aspecto (elongación)', lambda p: p.aspect_ratio),
            'compactness': ('Complejidad de la forma', lambda p: p.compactness),
            'hole_confidence': ('Confianza en agujero estructural', lambda p: p.hole_confidence),
            'hu1': ('Momento Hu1 - Dispersión', lambda p: p.hu_moments[0] if len(p.hu_moments) > 0 else 0),
            'hu2': ('Momento Hu2 - Elongación', lambda p: p.hu_moments[1] if len(p.hu_moments) > 1 else 0),
            'hu3': ('Momento Hu3 - Asimetría', lambda p: p.hu_moments[2] if len(p.hu_moments) > 2 else 0),
        }
        self._recommended_clustering_features = [
            'area',
            'solidity',
            'aspect_ratio',
            'hole_confidence',
            'hu1',
            'hu2',
            'hu3'
        ]
    
    def extract_features(self,
                         bounding_boxes: List[Tuple],
                         masks: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extrae características para todos los objetos detectados.
        
        Args:
            * bounding_boxes: Lista de bounding boxes
            * masks: Lista de máscaras binarias
            
        Returns:
            * Lista de diccionarios con características por objeto
        """
        if len(bounding_boxes) != len(masks):
            raise ValueError("El número de bounding boxes y máscaras debe coincidir")
        
        features_list = []
        
        # Para cada objeto, extraer características
        for i, (bbox, mask) in enumerate(zip(bounding_boxes, masks)):
            try:
                object_features = self._extract_single_object_features(bbox, mask, i + 1)
                features_list.append(object_features)
                
            except Exception as e:
                self.logger.error(f"Error extrayendo características del objeto {i}: {e}")
                continue
        
        self.logger.info(f"✅ Características extraídas para {len(features_list)} objetos")
        return features_list
    
    def _extract_single_object_features(self,
                                        bbox: Tuple,
                                        mask: np.ndarray,
                                        object_id: int) -> Dict[str, Any]:
        """
        Extrae características para un único objeto de manera eficiente.
        
        Args:
            * bbox: Bounding box del objeto (x, y, w, h)
            * mask: Máscara binaria del objeto
            * object_id: Identificador del objeto
            
        Returns:
            * Diccionario con todas las características calculadas
        """
        # 1. Calcular todas las propiedades geométricas una sola vez
        contour_manager = ContourManager(mask)
        properties = contour_manager.calculate_all_properties()

        # 2. Extraer características esenciales
        essential_features = self._extract_essential_features(properties)
        
        # 3. Combinar todas las características
        object_features = {
            'object_id': object_id,
            'bounding_box': bbox,
            **essential_features,
            # Características adicionales para análisis
            'hull_area': properties.hull_area,
            'has_structural_hole': properties.has_structural_hole,
            'moments': properties.moments
        }
        
        return object_features
    
    def _extract_essential_features(self, properties: GeometricProperties) -> Dict[str, float]:
        """
        Extrae características usando las definiciones centralizadas.
        ¡CERO duplicación de lógica!
        """
        return {
            feature_name: extractor_function(properties)
            for feature_name, (description, extractor_function) in self._feature_definitions.items()
        }
    
    def get_essential_feature_names(self) -> List[str]:
        """Nombres de todas las características disponibles"""
        return list(self._feature_definitions.keys())
    
    def get_recommended_features_for_clustering(self) -> List[str]:
        """Características recomendadas para clustering"""
        return self._recommended_clustering_features.copy()
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Descripciones detalladas de cada característica"""
        return {
            feature_name: description
            for feature_name, (description, _) in self._feature_definitions.items()
        }
    
    def extract_feature_matrix(self,
                               features_list: List[Dict],
                               feature_names: List[str] = None,
                               normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Convierte la lista de características en una matriz numpy para machine learning.
        
        Args:
            * features_list: Lista de diccionarios de características
            * feature_names: Nombres de características a incluir (None = automatico)
            * normalize: Si normalizar la matriz para ML (media 0, varianza 1)
            
        Returns:
            * Tuple: (matriz_features, nombres_características_usadas)
        """
        if feature_names is None:
            feature_names = self.get_recommended_features_for_clustering()
        
        matrix = []
        for features in features_list:
            row = [features[feature] for feature in feature_names]
            matrix.append(row)
        
        return np.array(matrix)
    
    def get_feature_statistics(self, features_list: List[Dict]) -> Dict[str, Dict]:
        """
        Provee estadísticas para que KMeans decida cómo normalizar.
        """
        feature_matrix, feature_names = self.extract_feature_matrix(features_list)
        
        stats = {}
        for i, feature_name in enumerate(feature_names):
            values = feature_matrix[:, i]
            stats[feature_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75)
            }
        
        return stats