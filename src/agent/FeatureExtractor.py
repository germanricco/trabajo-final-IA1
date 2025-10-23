import cv2
import numpy as np
from typing import List, Dict, Tuple
import math

class FeatureExtractor:
    def __init__(self):
        self.features_names = [
            'circularity',           # Qué tan circular es (arandelas ≈ 1.0)
            'elongation',            # Qué tan alargado (clavos > 3.0)
            'solidity',              # Presencia de huecos (arandelas < 0.8)
            'hexagonal_signature',   # Firma hexagonal (tuercas ≈ 0.8-1.0)
            'head_body_ratio',       # Relación cabeza-cuerpo (tornillos > 0.5)
            'thread_complexity',     # Complejidad de rosca (tornillos > 0.6)
            'perimeter_area_ratio',  # Complejidad del contorno
            'hu1_compactness',       # Momento Hu1 - Compactidad
            'hu2_elongation_hu',     # Momento Hu2 - Elongación
            'hu7_symmetry',          # Momento Hu7 - Simetría rotacional
            'num_vertices',          # Número de vértices poligonales
            'convexity_defects'      # Defectos de convexidad (huecos)
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

        # Obtener contorno principal
        contour = self._get_contour_from_mask(mask)
        if contour is None:
            return self._create_default_features(obj_id, bbox)
        
        # 1. Calcular ropiedades basicas
        basic_props = self._calculate_basic_properties(contour, mask)
        # Si hay algun problema, devolver caracteristicas por defecto (0)
        if basic_props["area"] == 0:
            return self._create_default_features(obj_id, bbox)
        
        #! DEBUG
        print(f"   ✅ Contorno principal: {basic_props['area']}")
        print(f"   ✅ Perímetro contorno: {basic_props['perimeter']:.2f}")
        print(f"   ✅ Área convex hull: {basic_props['hull_area']:.2f}")
        print(f"   ✅ Perímetro convex hull: {basic_props['hull_perimeter']:.2f}")
        print(f"   ✅ Momentos Hu: {basic_props['hu_moments']}")
        print(f"   ✅ Numero de vertices: {basic_props['num_vertices']}")
        
        # 2. Calcular propiedades avanzadas
        advanced_props = self._calculate_advanced_properties(contour, basic_props, bbox)
        print(f"")
        features = {
            "object_id": obj_id,
            "bounding_box": bbox,
            **basic_props,
            **advanced_props
        }
        
        self._print_optimized_analysis(obj_id, features)
        return features

    def _calculate_basic_properties(self, contour: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Calcula todas las propiedades básicas una sola vez
        """
        # Propiedades fundamentales
        area = float(np.sum(mask > 0))
        perimeter = float(cv2.arcLength(contour, True))
        
        # Convex hull y propiedades relacionadas
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        hull_perimeter = float(cv2.arcLength(hull, True))
        
        # Momentos de Hu
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)

        # Convertir momentos Hu a floats
        hu_moments_floats = []
        for i in range(7):
            hu_val = hu_moments[i][0] if hu_moments[i][0] != 0 else 0.0
            hu_moments_floats.append(float(hu_val))

        # Aproximación poligonal (se usa en múltiples características)
        poly_epsilon = 0.02 * perimeter
        approx_poly = cv2.approxPolyDP(contour, poly_epsilon, True)
        num_vertices = int(len(approx_poly))
        
        return {
            'area': area,
            'perimeter': perimeter,
            'hull_area': hull_area,
            'hull_perimeter': hull_perimeter,
            'hu_moments': hu_moments_floats,
            'num_vertices': num_vertices,
            'hull': hull
        }    


    def _calculate_advanced_properties(self, contour: np.ndarray,
                                    basic_props: Dict,
                                    bbox: Tuple) -> Dict:
        """
        Calcula propiedades avanzadas usando las propiedades básicas precomputadas
        """

        # Tomo todos las propiedades basicas
        area = basic_props['area']
        perimeter = basic_props['perimeter']
        hull_area = basic_props['hull_area']
        hull_perimeter = basic_props['hull_perimeter']
        hu_moments = basic_props['hu_moments']
        num_vertices = basic_props['num_vertices']
        hull = basic_props['hull']

        # 1. CIRCULARIDAD
        circularity = self._calculate_circularity(area, perimeter)

        # 2. ESBELTEZ
        elongation = self._calculate_elongation(contour)

        # 3. SOLIDEZ
        solidity = self._calculate_solidity(area, hull_area)
        
        # 4. FIRMA HEXAGONAL
        hexagonal_signature = self._calculate_hexagonal_signature(num_vertices, hu_moments)
        
        # 5. RELACIÓN CABEZA-CUERPO
        head_body_ratio = self._calculate_head_body_ratio(contour, elongation, bbox)
        
        # 6. COMPLEJIDAD DE ROSCA
        thread_complexity = self._calculate_thread_complexity(perimeter, hull_perimeter)
        
        # 7. RELACIÓN PERÍMETRO/ÁREA
        perimeter_area_ratio = perimeter / (area + 1e-5)
        
        # 8. MOMENTOS DE HU NORMALIZADOS
        hu_normalized = self._normalize_hu_moments(hu_moments)
        
        # 9. DEFECTOS DE CONVEXIDAD
        convexity_defects = self._calculate_convexity_defects(contour, hull, area)
        
        return {
            'circularity': circularity,
            'elongation': elongation,
            'solidity': solidity,
            'hexagonal_signature': hexagonal_signature,
            'head_body_ratio': head_body_ratio,
            'thread_complexity': thread_complexity,
            'perimeter_area_ratio': perimeter_area_ratio,
            'hu1_compactness': hu_normalized['hu1'],
            'hu2_elongation_hu': hu_normalized['hu2'],
            'hu7_symmetry': hu_normalized['hu7'],
            'convexity_defects': convexity_defects
        }


    def _get_contour_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Obtiene el contorno principal de la máscara"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Tomar el contorno con mayor área
        return max(contours, key=cv2.contourArea)
    
    def _calculate_circularity(self, area: float, perimeter: float) -> float:
        """Calcula circularidad"""
        if perimeter > 0:
            circularity = (4 * math.pi * area) / (perimeter ** 2)
        else:
            circularity = 0
        return max(0, min(1, circularity))
    
    def _calculate_elongation(self, contour: np.ndarray) -> float:
        """Calcula esbeltez usando el rectángulo de área mínima"""
        if len(contour) < 5:
            return 0.0
            
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        if min(width, height) == 0:
            return 0.0
            
        elongation = max(width, height) / min(width, height)
        return min(elongation / 10, 1.0)  # Normalizar a [0,1]


    def _calculate_solidity(self, area: float, hull_area: float) -> float:
        """Calcula solidez"""
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 1.0
        return max(0.0, min(1.0, solidity))


    def _calculate_hexagonal_signature(self, num_vertices: int, hu_moments: np.ndarray) -> float:
        """Calcula firma hexagonal usando vértices precomputados y momentos Hu"""
        # Puntaje basado en número de vértices
        if num_vertices == 6:
            vertex_score = 1.0
        elif num_vertices in [5, 7]:
            vertex_score = 0.7
        elif num_vertices in [4, 8]:
            vertex_score = 0.4
        else:
            vertex_score = 0.0
        
        # Análisis de simetría usando momentos Hu precomputados
        hu7 = hu_moments[6]
        if abs(hu7) > 1e-5:
            symmetry = 1.0 - min(1.0, abs(hu7) / 10)
        else:
            symmetry = 0.0
        
        return (vertex_score + symmetry) / 2
    
    def _calculate_head_body_ratio(self, contour: np.ndarray, elongation: float, bbox: Tuple) -> float:
        """Calcula relación entre cabeza y cuerpo para tornillos/clavos"""
        if elongation < 0.5:  # No es alargado
            return 0.0
            
        # Para objetos alargados, estimar cabeza vs cuerpo
        # (Implementación simplificada - puedes mejorarla)
        x, y, w, h = bbox
        aspect_ratio = max(w, h) / min(w, h)
        
        # Tornillos tienen cabeza más ancha en relación al cuerpo
        if aspect_ratio > 3:
            return 0.3  # Probable clavo
        elif aspect_ratio > 2:
            return 0.6  # Probable tornillo
        else:
            return 0.1

    def _calculate_thread_complexity(self, perimeter: float, hull_perimeter: float) -> float:
        """Calcula complejidad de rosca usando perímetros precomputados"""
        if hull_perimeter == 0:
            return 0.0
            
        complexity_ratio = perimeter / hull_perimeter
        return max(0, min(1, (complexity_ratio - 1.0) * 5))

    def _normalize_hu_moments(self, hu_moments: np.ndarray) -> Dict[str, float]:
        """Normaliza momentos Hu"""
        hu_normalized = {}
        important_indices = [0, 1, 6]  # Hu1, Hu2, Hu7
        
        for i in important_indices:
            hu_val = hu_moments[i]
            if abs(hu_val) > 1e-5:
                normalized = -1 * math.copysign(1, hu_val) * math.log10(abs(hu_val))
            else:
                normalized = 0.0
            
            names = {0: 'hu1', 1: 'hu2', 6: 'hu7'}
            hu_normalized[names[i]] = normalized
        
        return hu_normalized
    
    def _calculate_convexity_defects(self, contour: np.ndarray, hull: np.ndarray, area: float) -> float:
        """Calcula defectos de convexidad usando hull"""
        if len(contour) < 10 or area == 0:
            return 0.0
            
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if len(hull_indices) < 4:
                return 0.0
                
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is None:
                return 0.0
                
            total_depth = sum(defect[0, 3] for defect in defects) / 256.0
            return total_depth / area
            
        except:
            return 0.0
    
    def _create_default_features(self, obj_id: int, bbox: Tuple) -> Dict:
        """Crea características por defecto cuando no hay contorno"""
        default_features = {
            'object_id': obj_id,
            'bounding_box': bbox,
            'area': 0.0,
            'perimeter': 0.0,
            'hull_area': 0.0,
            'hull_perimeter': 0.0,
            'hu_moments': [0.0] * 7,
            'num_vertices': 0,
            'circularity': 0.0,
            'elongation': 0.0,
            'solidity': 0.0,
            'hexagonal_signature': 0.0,
            'head_body_ratio': 0.0,
            'thread_complexity': 0.0,
            'perimeter_area_ratio': 0.0,
            'hu1_compactness': 0.0,
            'hu2_elongation_hu': 0.0,
            'hu7_symmetry': 0.0,
            'convexity_defects': 0.0
        }
        return default_features
    
    def _print_optimized_analysis(self, obj_id: int, features: Dict):
        """Análisis optimizado sin cálculos redundantes"""
        print(f"   ✅ Pieza {obj_id} - Características OPTIMIZADAS:")
        print(f"      🔴 Circularidad:       {features['circularity']:.3f}")
        print(f"      📏 Esbeltez:           {features['elongation']:.3f}")
        print(f"      🏢 Solidez:            {features['solidity']:.3f}")
        print(f"      🔩 Firma hexagonal:    {features['hexagonal_signature']:.3f}")
        print(f"      ⚖️  Relación cabeza:    {features['head_body_ratio']:.3f}")
        print(f"      🧵 Complejidad rosca:  {features['thread_complexity']:.3f}")
    
    
    def get_feature_vector(self, features_list: List[Dict]) -> np.ndarray:
        """Convierte a vector para machine learning"""
        vectors = []
        for features in features_list:
            vector = [features[name] for name in self.feature_names]
            vectors.append(vector)
        return np.array(vectors)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names