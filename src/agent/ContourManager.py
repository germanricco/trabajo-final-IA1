import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class GeometricProperties:
    """Almacena todas las propiedades geométricas calculadas una sola vez"""
    area: float
    perimeter: float
    hull_area: float
    convex_hull: np.ndarray
    min_area_rect: Tuple
    aspect_ratio: float
    circularity: float
    compactness: float
    solidity: float
    moments: Dict[str, float]
    hu_moments: np.ndarray
    has_structural_hole: bool
    hole_confidence: float
    external_contour: np.ndarray
    internal_contour: Optional[np.ndarray]

class ContourManager:
    """
    Gestiona los contornos y calcula todas las propiedades geométricas
    """
    
    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(np.uint8)
        self._properties: Optional[GeometricProperties] = None
        self._calculated = False
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_properties(self) -> GeometricProperties:
        """
        Calcula todas las propiedades geométricas de una vez.
        """
        # Si ya estan calculadas, retornarlas
        if self._calculated and self._properties is not None:
            return self._properties
        
        try:
            # 1. Encontrar contornos externos e internos
            external_contour, internal_contour = self._find_contours_hierarchical()
            
            if external_contour is None:
                raise ValueError("No se pudo encontrar contorno externo")
            
            # 2. Calcular propiedades básicas
            area = cv2.contourArea(external_contour)
            perimeter = cv2.arcLength(external_contour, True)
            
            # 3. Calcular casco convexo y propiedades derivadas
            convex_hull = cv2.convexHull(external_contour)
            hull_area = cv2.contourArea(convex_hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 4. Calcular rectángulo de área mínima y relación de aspecto
            min_area_rect = cv2.minAreaRect(external_contour)
            width, height = min_area_rect[1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            # 5. Calcular circularidad y compacidad
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            compactness = (perimeter ** 2) / area if area > 0 else 0
            
            # 6. Calcular momentos y momentos de Hu
            moments = cv2.moments(external_contour)
            hu_moments_raw = cv2.HuMoments(moments).flatten()
            hu_moments = self._normalize_hu_moments(hu_moments_raw)
            
            # 7. Clasificar agujeros estructurales
            has_structural_hole, hole_confidence = self._classify_structural_hole(
                external_contour, internal_contour, area
            )
            
            # Almacenar todas las propiedades calculadas
            self._properties = GeometricProperties(
                area=area,
                perimeter=perimeter,
                hull_area=hull_area,
                convex_hull=convex_hull,
                min_area_rect=min_area_rect,
                aspect_ratio=aspect_ratio,
                circularity=circularity,
                compactness=compactness,
                solidity=solidity,
                moments=self._extract_key_moments(moments),
                hu_moments=hu_moments,
                has_structural_hole=has_structural_hole,
                hole_confidence=hole_confidence,
                external_contour=external_contour,
                internal_contour=internal_contour
            )
            
            self._calculated = True
            return self._properties
            
        except Exception as e:
            self.logger.error(f"Error calculando propiedades: {e}")
            raise
    
    def _find_contours_hierarchical(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Encuentra contornos externos e internos usando jerarquía.
        """
        contours, hierarchy = cv2.findContours(
            self.mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, None
        
        # Encontrar el contorno externo más grande
        external_contours = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # Contorno externo (sin padre)
                external_contours.append(contour)
        
        if not external_contours:
            return None, None
        
        external_contour = max(external_contours, key=cv2.contourArea)
        
        # Buscar contornos internos (hijos del contorno externo)
        internal_contour = None
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == 0:  # Hijo directo del primer contorno externo
                if internal_contour is None or cv2.contourArea(contour) > cv2.contourArea(internal_contour):
                    internal_contour = contour
        
        return external_contour, internal_contour
    
    def _normalize_hu_moments(self, hu_moments: np.ndarray) -> np.ndarray:
        """
        Normaliza momentos de Hu usando transformación logarítmica.
        """
        # Transformación logarítmica para comprimir rango dinámico
        hu_log = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)
        
        # Retornar solo los momentos más discriminativos (Hu1, Hu2, Hu3)
        return hu_log[:3]  # Solo los primeros 3 momentos
    
    def _extract_key_moments(self, moments: Dict) -> Dict[str, float]:
        """
        Extrae los momentos más importantes del diccionario de OpenCV.
        """
        return {
            'm00': moments.get('m00', 0),
            'm10': moments.get('m10', 0),
            'm01': moments.get('m01', 0),
            'mu20': moments.get('mu20', 0),
            'mu11': moments.get('mu11', 0),
            'mu02': moments.get('mu02', 0)
        }
    
    def _classify_structural_hole(self, external_contour: np.ndarray, 
                                internal_contour: Optional[np.ndarray], 
                                external_area: float) -> Tuple[bool, float]:
        """
        Clasifica si existe un agujero estructural usando múltiples características.
        """
        # Si no hay contorno interno, no hay agujero estructural
        if internal_contour is None:
            return False, 0.0
        
        # Calcular características del agujero
        hole_area = cv2.contourArea(internal_contour)
        hole_perimeter = cv2.arcLength(internal_contour, True)
        
        # 1. Relación de área del agujero
        hole_area_ratio = hole_area / external_area if external_area > 0 else 0
        
        # 2. Circularidad del agujero
        hole_circularity = (4 * np.pi * hole_area) / (hole_perimeter ** 2) if hole_perimeter > 0 else 0
        
        # 3. Centrado del agujero
        M_ext = cv2.moments(external_contour)
        M_int = cv2.moments(internal_contour)
        
        if M_ext["m00"] != 0 and M_int["m00"] != 0:
            cX_ext = M_ext["m10"] / M_ext["m00"]
            cY_ext = M_ext["m01"] / M_ext["m00"]
            cX_int = M_int["m10"] / M_int["m00"]
            cY_int = M_int["m01"] / M_int["m00"]
            
            centroid_distance = np.sqrt((cX_ext - cX_int)**2 + (cY_ext - cY_int)**2)
            equivalent_diameter = np.sqrt(4 * external_area / np.pi)
            normalized_distance = centroid_distance / equivalent_diameter
        else:
            normalized_distance = 1.0
        
        # Sistema de puntuación
        score = 0.0
        
        # Puntuación por tamaño
        if hole_area_ratio > 0.15:
            score += 2.0
        elif hole_area_ratio > 0.05:
            score += 1.0
        
        # Puntuación por circularidad
        if hole_circularity > 0.7:
            score += 2.0
        elif hole_circularity > 0.5:
            score += 1.0
        
        # Puntuación por centrado
        if normalized_distance < 0.1:
            score += 2.0
        elif normalized_distance < 0.2:
            score += 1.0
        
        # Umbral de decisión
        HOLE_CONFIDENCE_THRESHOLD = 3.5
        has_structural_hole = score >= HOLE_CONFIDENCE_THRESHOLD
        confidence = min(score / HOLE_CONFIDENCE_THRESHOLD, 1.0)
        
        return has_structural_hole, confidence
    
    @property
    def properties(self) -> GeometricProperties:
        """Acceso seguro a las propiedades calculadas"""
        if not self._calculated:
            return self.calculate_all_properties()
        return self._properties