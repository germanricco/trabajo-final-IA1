import cv2
import numpy as np
from typing import List, Dict, Tuple
import math

class FeatureExtractor:
    def __init__(self):
        self.features_names = ['area', 'circularity', 'aspect_ratio']

    def extract_features(self, original_image: np.ndarray,
                         bounding_boxes: List[Tuple],
                         masks: List[np.ndarray]) -> List[Dict]:

        """
        Extrae caracteristicas para cada objeto detectado

        Args:
            original_image (np.ndarray): Imagen original
            bounding_boxes (List[Tuple]): Lista de cajas delimitadoras
            masks (List[np.ndarray]): Lista de mascaras

        Returns:
            List[Dict]: Lista de diccionarios con las caracteristicas
        """

        features_list = []

        for i, (bbox, mask) in enumerate(zip(bounding_boxes, masks)):
            print(f"🔍 Analizando objeto {i+1}...")
            
            # Extraer características individuales
            features = self._extract_single_object_features(original_image, bbox, mask, i+1)
            features_list.append(features)
        
        return features_list

    def _extract_single_object_features(self, original_image: np.ndarray,
                                      bbox: Tuple, 
                                      mask: np.ndarray,
                                      obj_id: int) -> Dict:
        """Extrae características para un solo objeto"""
        x, y, w, h = bbox
        
        # 1. ÁREA - Número de píxeles en la máscara
        area = np.sum(mask > 0)
        
        # 2. ESBELTEZ - Relación de aspecto del bounding box
        aspect_ratio = self._calculate_aspect_ratio(w, h)
        
        # 3. REDONDEZ - Qué tan circular es el objeto
        circularity = self._calculate_circularity(mask)
        
        # Crear diccionario de características
        features = {
            'object_id': obj_id,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'bounding_box': bbox
        }
        
        # Mostrar resultados en consola
        self._print_features(obj_id, features)
        
        return features
    
    def _calculate_aspect_ratio(self, width: float, height: float) -> float:
        """Calcula la relación de aspecto (esbeltez)"""
        if min(width, height) == 0:
            return 0
        return max(width, height) / min(width, height)
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calcula la redondez usando la fórmula de circularidad"""
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Usar el contorno más grande
        contour = max(contours, key=cv2.contourArea)
        
        # Calcular área y perímetro
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Fórmula de circularidad: (4 * π * área) / (perímetro²)
        if perimeter > 0:
            circularity = (4 * math.pi * area) / (perimeter ** 2)
        else:
            circularity = 0.0
        
        return circularity
    
    def _print_features(self, obj_id: int, features: Dict):
        """Muestra las características en la consola"""
        print(f"   Objeto {obj_id}:")
        print(f"     📐 Área: {features['area']} píxeles")
        print(f"     📏 Esbeltez: {features['aspect_ratio']:.2f}")
        print(f"     🔴 Redondez: {features['circularity']:.3f}")

    
    def visualize_features(original_image: np.ndarray, 
                      features_list: List[Dict],
                      bounding_boxes: List[Tuple]):
        """Visualiza los objetos con sus características"""
        # Crear imagen para visualización
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:  # Si es escala de grises, convertir a color
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        for features, bbox in zip(features_list, bounding_boxes):
            obj_id = features['object_id']
            x, y, w, h = bbox
            
            # Dibujar bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Texto con características
            text_lines = [
                f"Obj {obj_id}",
                f"Area: {features['area']}",
                f"Esbeltez: {features['aspect_ratio']:.1f}",
                f"Redondez: {features['circularity']:.2f}"
            ]
            
            # Posición del texto (arriba del bounding box)
            text_y = max(y - 10, 20)
            for i, line in enumerate(text_lines):
                y_pos = text_y + i * 20
                cv2.putText(vis_image, line, (x, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Redimensionar si es muy grande
        h, w = vis_image.shape[:2]
        if w > 800:
            vis_image = cv2.resize(vis_image, (800, int(800 * h / w)))
        
        cv2.imshow('Características Extraídas', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Función de prueba completa
    def test_feature_extractor(original_image: np.ndarray, 
                            segmentator_result: Dict):
        """
        Función completa para probar el FeatureExtractor
        """
        print("🚀 INICIANDO EXTRACCIÓN DE CARACTERÍSTICAS")
        print("=" * 50)
        
        # Extraer datos del resultado del segmentador
        bounding_boxes = segmentator_result["bounding_boxes"]
        masks = segmentator_result["masks"]
        
        print(f"📦 Objetos a analizar: {len(bounding_boxes)}")
        
        # Crear y usar el FeatureExtractor
        extractor = FeatureExtractor()
        features_list = extractor.extract_features(original_image, bounding_boxes, masks)
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE CARACTERÍSTICAS:")
        print("=" * 30)
        for features in features_list:
            obj_id = features['object_id']
            print(f"Objeto {obj_id}: Area={features['area']}, "
                f"Esbeltez={features['aspect_ratio']:.2f}, "
                f"Redondez={features['circularity']:.3f}")
        
        # Visualizar resultados
        #print("\n🎨 Visualizando resultados...")
        #visualize_features(original_image, features_list, bounding_boxes)
        
        return features_list