import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from src.agent.ContourManager import ContourManager

class FeatureExtractor:
    """
    Puente entre el análisis geométrico (ContourManager) y el modelo de ML.
    Selecciona y aplana las características más relevantes para la clasificación.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Definimos explícitamente qué características usaremos para el clustering
        # El orden aquí es importante si no usaras DataPreprocessor, pero 
        # con él, nos aseguramos consistencia por nombres.
        self.CLUSTERING_FEATURES = [
            'area', 
            'perimeter', 
            'solidity', 
            'circularity', 
            'aspect_ratio', 
            'hole_confidence',
            'circle_ratio',
            'num_vertices',
            'hu1', 'hu2', 'hu3' # Momentos invariantes
        ]

    def extract_features(self, bounding_boxes: List[Tuple], masks: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Procesa una lista de objetos detectados y retorna sus características.
        """
        if len(bounding_boxes) != len(masks):
            raise ValueError(f"Desajuste: {len(bounding_boxes)} bboxes vs {len(masks)} máscaras.")

        features_list = []

        for i, (bbox, mask) in enumerate(zip(bounding_boxes, masks)):
            try:
                # 1. Delegar matemática pesada al ContourManager
                manager = ContourManager(mask)
                props = manager.calculate_all_properties()

                # 2. Aplanar datos para el dataset
                obj_data = self._map_properties_to_dict(props, i)
                
                # 3. Agregar contexto útil para la UI (no para el KMeans)
                obj_data['bbox'] = bbox 
                
                features_list.append(obj_data)

            except Exception as e:
                self.logger.warning(f"Omitiendo objeto {i} por error de cálculo: {e}")
                continue

        return features_list

    def _map_properties_to_dict(self, p: Any, obj_id: int) -> Dict[str, float]:
        """
        Transforma el objeto GeometricProperties en un diccionario plano.
        Aquí es donde 'elegimos' qué datos le importan a la IA.
        """
        # Seguridad para Momentos de Hu (por si el contorno es degenerado)
        hu = p.hu_moments if hasattr(p, 'hu_moments') and len(p.hu_moments) >= 3 else [0, 0, 0]

        return {
            # Metadatos
            'id': obj_id,
            
            # Características Geométricas Básicas
            'area': float(p.area),
            'perimeter': float(p.perimeter),
            'solidity': float(p.solidity),
            'circularity': float(p.circularity),
            'aspect_ratio': float(p.aspect_ratio),
            'compactness': float(p.compactness),
            
            # Características Estructurales
            'hole_confidence': float(p.hole_confidence), # Clave para tuercas/arandelas
            'circle_ratio': float(p.circle_ratio),     # Clave para distinguir círculos
            'num_vertices': float(p.num_vertices), # Clave para distinguir formas
            
            # Momentos de Hu (Invariantes a rotación)
            'hu1': float(hu[0]), # Dispersión
            'hu2': float(hu[1]), # Elongación
            'hu3': float(hu[2]), # Asimetría
        }

    def get_recommended_features(self) -> List[str]:
        """Retorna las claves que deberían usarse para el entrenamiento"""
        return self.CLUSTERING_FEATURES