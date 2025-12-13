import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from src.vision.contours import ContourManager

class FeatureExtractor:
    """
    Puente entre el análisis geométrico (ContourManager) y el modelo de ML.
    Selecciona y aplana las características más relevantes para la clasificación.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Definimos qué características usaremos para el clustering
        self.CLUSTERING_FEATURES = [
            'aspect_ratio',
            'solidity', 
            'hole_confidence',
            'circle_ratio',
            'radius_variance',
        ]

    def extract_features(self, bounding_boxes: List[Tuple], masks: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Procesa una lista de objetos detectados y retorna sus características.

        Args:
            * bounding_boxes (List[Tuple]): Lista de bounding boxes (x, y, w, h).
            * masks (List[np.ndarray]): Lista de máscaras binarias correspondientes.

        Returns:
            * List[Dict[str, Any]]: Lista de diccionarios con las características extraidas.
        """
        # Validacion básica
        if len(bounding_boxes) != len(masks):
            self.logger.warning(f"Desajuste: {len(bounding_boxes)} bboxes y {len(masks)} masks.")
            return []
        
        features_list = []

        for i, (bbox, mask) in enumerate(zip(bounding_boxes, masks)):
            try:
                # Delegar matemática pesada al ContourManager
                manager = ContourManager(mask)
                props = manager.calculate_all_properties()

                if props is None:
                    continue

                # Aplanar datos para el dataset
                obj_data = self._map_properties_to_dict(props, i)
                
                # Agregar contexto útil para la UI (no para el KMeans)
                obj_data['bbox'] = bbox 
                
                features_list.append(obj_data)

            except Exception as e:
                self.logger.warning(f"Omitiendo objeto {i} por error de cálculo: {e}")
                continue

        return features_list

    def _map_properties_to_dict(self, p: Any, obj_id: int) -> Dict[str, float]:
        """
        Transforma el objeto GeometricProperties en un diccionario plano.
        """
        # Seguridad para Momentos de Hu (por si el contorno es degenerado)
        hu = p.hu_moments if hasattr(p, 'hu_moments') and len(p.hu_moments) >= 3 else [0.0]*7

        return {
            # --- METADATOS ---
            'id': obj_id,
            
            # --- 1. FEATURES DE CLUSTERING (THE BIG 5) ---
            # Estas son las únicas que el K-Means mirará (si configuramos bien el FeatureExtractor)
            'hole_confidence': float(p.hole_confidence),  # Tuerca/Arandela vs Resto
            'aspect_ratio': float(p.aspect_ratio),        # Largo vs Corto
            'roughness': float(p.roughness),              # Rosca vs Liso -> Tornillo vs Clavo
            'radius_variance': float(p.radius_variance),  # Hexágono vs Círculo -> Tuerca vs Arandela
            
            
            # --- 2. FEATURES INFORMATIVAS (DEBUG / LEGACY) ---
            # Se guardan para visualización en tablas o análisis futuro, 
            # pero no dirigirán la clasificación principal.
            'rectangularity': float(p.rectangularity), 
            'solidity': float(p.solidity),          # Reemplazado por roughness
            'circle_ratio': float(p.circle_ratio),  # Reemplazado por radius_variance
            'extent': float(p.extent),              # Similar a rectangularity pero sin rotación
            'circularity': float(p.circularity),    # Muy sensible al ruido
            'num_vertices': float(p.num_vertices),  # Inestable
            
            # Datos físicos crudos
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