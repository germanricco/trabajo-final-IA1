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
            'aspect_ratio',
            'extent',
            'solidity', 
            'circularity',
            'hole_confidence',
            'circle_ratio',
            'radius_variance',
            #'num_vertices',
        ]

    def extract_features(self, bounding_boxes: List[Tuple], masks: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Procesa una lista de objetos detectados y retorna sus características.
        """
        if len(bounding_boxes) != len(masks):
            self.logger.warning(f"Desajuste: {len(bounding_boxes)} bboxes y {len(masks)} masks.")
            return []
        
        features_list = []

        for i, (bbox, mask) in enumerate(zip(bounding_boxes, masks)):
            try:
                # 1. Delegar matemática pesada al ContourManager
                manager = ContourManager(mask)
                props = manager.calculate_all_properties()

                if props is None:
                    continue

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
        hu = p.hu_moments if hasattr(p, 'hu_moments') and len(p.hu_moments) >= 3 else [0.0]*7

        return {
            # Metadatos
            'id': obj_id,
            
            # --- FEATURES DE CLUSTERING (PRIORITARIAS) ---
            'aspect_ratio': float(p.aspect_ratio),
            'extent': float(p.extent),
            'solidity': float(p.solidity),
            'circularity': float(p.circularity),
            'hole_confidence': float(p.hole_confidence),
            'circle_ratio': float(p.circle_ratio),
            'radius_variance': float(p.radius_variance),
            'num_vertices': float(p.num_vertices),
            
            # --- FEATURES INFORMATIVAS (DEBUG/UI) ---
            'area': float(p.area),
            'perimeter': float(p.perimeter),
            'compactness': float(p.compactness),
            'hu1': float(hu[0]),
            'hu2': float(hu[1]),
            'hu3': float(hu[2]),
        }

    def get_recommended_features(self) -> List[str]:
        """Retorna las claves que deberían usarse para el entrenamiento"""
        return self.CLUSTERING_FEATURES