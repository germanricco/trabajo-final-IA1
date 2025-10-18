import cv2
import numpy as np
from typing import List, Dict, Tuple
import math

class FeatureExtractor:
    def __init__(self):
        self.features_names = [
            'hu1_compactness',      # Compacidad
            'hu2_elongation',       # Elongacion/simetria
            'hu7_symmetry',         # Simetria rotacional
            'perimeter_area_ratio', # Relacion de perimetro y area
            'convexity_ratio',      # Presencia de huecos
            'aspect_ratio'          # Esbeltez
        ]

    def extract_features(self,
                         bounding_boxes: List[Tuple],
                         masks: List[np.ndarray]) -> List[Dict]:

        """
        Extrae caracteristicas para cada objeto detectado

        Args:
            bounding_boxes (List[Tuple]): Lista de cajas delimitadoras
            masks (List[np.ndarray]): Lista de mascaras

        Returns:
            List[Dict]: Lista de diccionarios con las caracteristicas
        """

        features_list = []

        print(f"🚀 INICIANDO EXTRACCION DE CARACTERISTICAS")
        for i, (bbox, mask) in enumerate(zip(bounding_boxes, masks)):
            print(f"   🔍 Analizando objeto {i+1}...")
            
            # Extraer características individuales
            features = self._extract_single_object_features(bbox, mask, i+1)
            features_list.append(features)
        
        return features_list

    def _extract_single_object_features(self,
                                      bbox: Tuple, 
                                      mask: np.ndarray,
                                      obj_id: int) -> Dict:
        """Extrae características para un solo objeto"""
        x, y, w, h = bbox
        
        # Caracteristicas Basicas necesarias
        area = np.sum(mask > 0)
        perimeter = self._calculate_perimeter(mask)
        
        # Caracteristicas Clave
        hu_moments = self._calculate_discriminative_hu(mask)
        perimeter_area_ratio = perimeter / (area + 1e-5)
        convexity_ratio = self._calculate_convexity_ratio(mask)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # Almaceno en directorio
        features = {
            'object_id': obj_id,
            'hu1_compactness': hu_moments['hu1'],
            'hu2_elongation': hu_moments['hu2'],
            'hu7_symmetry': hu_moments['hu7'],
            'perimeter_area_ratio': perimeter_area_ratio,
            'convexity_ratio': convexity_ratio,
            'aspect_ratio': aspect_ratio,
            'bounding_box': bbox
        }
        
        #! Debuggeando imprimo en pantalla 
        self._print_optimized_features(obj_id, features)

        return features
    

    def _calculate_perimeter(self, mask:np.ndarray) -> float:
        """Calcula el perimetro desde la mascara"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.arcLength(contours[0], True) if contours else 0.0

    def _calculate_convexity_ratio(self, mask: np.ndarray) -> float:
        """Calcula relación de convexidad (área / área_convex_hull)"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        contour = contours[0]
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        return area / hull_area if hull_area > 0 else 0.0
    
    def _calculate_discriminative_hu(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calcula solo los momentos de Hu más discriminativos
        Hu1, Hu2, Hu7 son los más útiles para formas geométricas
        """
        moments = cv2.moments(mask)
        hu_moments = cv2.HuMoments(moments)
        
        # Solo los momentos más discriminativos
        discriminative_indices = [0, 1, 6]  # Hu1, Hu2, Hu7
        hu_normalized = {}
        
        for i in discriminative_indices:
            hu_val = hu_moments[i][0]
            if abs(hu_val) > 1e-5:
                normalized = -1 * math.copysign(1, hu_val) * math.log10(abs(hu_val))
            else:
                normalized = 0.0
            hu_name = ['hu1', 'hu2', 'hu7'][discriminative_indices.index(i)]
            hu_normalized[hu_name] = normalized
        
        return hu_normalized


    def _print_optimized_features(self, obj_id: int, features: Dict):
        """Muestra solo las características clave"""
        print(f"🔍 Objeto {obj_id}:")
        print(f"   📊 Compactidad (Hu1): {features['hu1_compactness']:7.3f}")
        print(f"   📏 Elongación (Hu2):  {features['hu2_elongation']:7.3f}")
        print(f"   🔄 Simetría (Hu7):    {features['hu7_symmetry']:7.3f}")
        print(f"   📐 Relación P/A:      {features['perimeter_area_ratio']:7.3f}")
        print(f"   🏢 Convexidad:        {features['convexity_ratio']:7.3f}")
        print(f"   📏 Esbeltez:          {features['aspect_ratio']:7.3f}")
        
        # Análisis rápido de cluster probable
        cluster_hint = self._cluster_analysis(features)
        print(f"   🎯 Cluster probable:  {cluster_hint}")
        print()
    
    def _cluster_analysis(self, features: Dict) -> str:
        """Análisis simple basado en las características clave"""
        hu1 = features['hu1_compactness']
        hu2 = features['hu2_elongation'] 
        hu7 = features['hu7_symmetry']
        aspect_ratio = features['aspect_ratio']
        convexity = features['convexity_ratio']
        perimeter_area = features['perimeter_area_ratio']
        
        # Lógica de clusterización (basada en patrones típicos)
        if aspect_ratio > 3.0:
            return "CLUSTER TORNILLOS/CLAVOS (muy alargados)"
        elif hu1 < -4.5 and convexity < 0.8 and perimeter_area < 0.1:
            return "CLUSTER ARANDELAS (redondas con hueco)"
        elif hu1 < -4.0 and convexity > 0.85 and hu7 > -12:
            return "CLUSTER TUERCAS (regulares, alta convexidad)"
        elif aspect_ratio < 2.0 and hu2 > -8:
            return "CLUSTER FORMAS COMPACTAS"
        else:
            return "CLUSTER FORMAS MIXTAS"
    
    def get_feature_vector(self, features_list: List[Dict]) -> np.ndarray:
        """Convierte a vector para K-Means"""
        vectors = []
        for features in features_list:
            vector = [features[name] for name in self.feature_names]
            vectors.append(vector)
        return np.array(vectors)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names
