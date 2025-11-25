from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

@dataclass
class InternalContourCandidate:
    """Almacena todas las propiedades de un contorno interno candidato"""
    contour: np.ndarray
    area: float = 0.0
    perimeter: float = 0.0
    circularity: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)
    normalized_distance: float = 0.0
    area_ratio: float = 0.0
    score: float = 0.0
    
    def calculate_properties(self, external_contour: np.ndarray, external_area: float) -> None:
        """
        Calcula TODAS las propiedades del contorno candidato de una sola vez.
        """
        try:
            # 1. Área y perímetro (básicos)
            self.area = cv2.contourArea(self.contour)
            self.perimeter = cv2.arcLength(self.contour, True)
            
            # 2. Circularidad
            self.circularity = (4 * np.pi * self.area) / (self.perimeter ** 2) if self.perimeter > 0 else 0
            
            # 3. Centroide
            M = cv2.moments(self.contour)
            if M["m00"] != 0:
                self.centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            else:
                self.centroid = (0, 0)
            
            # 4. Distancia normalizada al centro del contorno externo
            M_ext = cv2.moments(external_contour)
            if M_ext["m00"] != 0:
                cX_ext = M_ext["m10"] / M_ext["m00"]
                cY_ext = M_ext["m01"] / M_ext["m00"]
                
                centroid_distance = np.sqrt((self.centroid[0] - cX_ext)**2 + (self.centroid[1] - cY_ext)**2)
                equivalent_diameter = np.sqrt(4 * external_area / np.pi)
                self.normalized_distance = centroid_distance / equivalent_diameter
            else:
                self.normalized_distance = 1.0
            
            # 5. Relación de área
            self.area_ratio = self.area / external_area if external_area > 0 else 0
            
        except Exception as e:
            print(f"Error calculando propiedades del contorno candidato: {e}")