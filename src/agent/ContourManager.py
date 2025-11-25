import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from src.agent.InternalContourCandidate import InternalContourCandidate

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
    hu_moments: np.ndarray
    has_structural_hole: bool
    hole_confidence: float
    external_contour: np.ndarray
    internal_contour: Optional[np.ndarray]

class ContourManager:
    """
    Gestiona los contornos
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
            external_contour, hole_candidate = self._find_contours_hierarchical()
            
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
            
            # 7. Clasificar agujeros estructurales (internos)
            has_structural_hole, hole_confidence = self._classify_structural_hole(hole_candidate)
            
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
                hu_moments=hu_moments,
                has_structural_hole=has_structural_hole,
                hole_confidence=hole_confidence,
                external_contour=external_contour,
                internal_contour=hole_candidate.contour if hole_candidate else None
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
        
        # Encontrar el contorno externo principal
        external_contours = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # Contorno externo
                external_contours.append(contour)
        
        if not external_contours:
            return None, None
        
        external_contour = max(external_contours, key=cv2.contourArea)
        external_area = cv2.contourArea(external_contour)
        
        # Encontrar Indice del contorno externo principal
        external_contour_index = None
        for i, contour in enumerate(contours):
            if np.array_equal(contour, external_contour):
                external_contour_index = i
                break
        
        if external_contour_index is None:
            return external_contour, None
        
        # Buscar contornos internos (hijos del contorno externo)
        candidates = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == external_contour_index:  # Hijo del contorno externo
                area = cv2.contourArea(contour)

                # Filtro
                if area < 10 or area > 0.5 * external_area:
                    continue

                # Crear candidato y calcular propiedades
                candidate = InternalContourCandidate(contour=contour)
                candidate.calculate_properties(external_contour, external_area)

                # Calcular puntuacion usando propiedades
                candidate.score = self._calculate_internal_contour_score(
                    candidate.area_ratio,
                    candidate.circularity,
                    candidate.normalized_distance
                )

                candidates.append(candidate)
        
        # Si no hay candidatos para contorno interno, retornar None
        if not candidates:
            return external_contour, None
        
        # Seleccionar el mejor candidato basado en la puntuación
        best_candidate = max(candidates, key=lambda x: x.score)

        # Umbral de calidad
        MINIMUM_SCORE_THRESHOLD = 0.5
        if best_candidate.score >= MINIMUM_SCORE_THRESHOLD:
            return external_contour, best_candidate
        else:
            return external_contour, None
    
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
    
    def _classify_structural_hole(self, hole_candidate: Optional[InternalContourCandidate]) -> Tuple[bool, float]:
        """
        Clasifica si existe un agujero estructural usando múltiples características.
        """
        
        if hole_candidate is None:
            return False, 0.0
        
        try:
            # Usar propiedades YA CALCULADAS del candidato
            hole_area_ratio = hole_candidate.area_ratio
            hole_circularity = hole_candidate.circularity
            normalized_distance = hole_candidate.normalized_distance

            # Sistema de puntuación
            score = 0.0
            
            # Puntuación por tamaño
            if hole_area_ratio > 0.15:
                score += 2.0
            elif hole_area_ratio > 0.1:
                score += 1.0
            
            # Puntuación por circularidad
            if hole_circularity > 0.9:
                score += 3.0
            elif hole_circularity > 0.7:
                score += 2.0
            elif hole_circularity > 0.5:
                score += 1.0
            
            # Puntuación por centrado
            if normalized_distance < 0.1:
                score += 3.0
            elif normalized_distance < 0.15:
                score += 2.0
            elif normalized_distance < 0.2:
                score += 1.0

            # print(f"hole_area_ratio = {hole_area_ratio}")
            # print(f"hole_circularity = {hole_circularity}")
            # print(f"normalized_distance = {normalized_distance}")
            # print(f"hole score = {score}")

            # Umbral de decisión
            HOLE_CONFIDENCE_THRESHOLD = 4.5
            has_structural_hole = score >= HOLE_CONFIDENCE_THRESHOLD
            hole_confidence = min(score / HOLE_CONFIDENCE_THRESHOLD, 1.0)
        
            return has_structural_hole, hole_confidence
        
        except Exception as e:
            print(f"Error en _classify_structural_hole: {e}")
            return False, 0.0
        
    def _calculate_internal_contour_score(self, area_ratio: float, circularity: float, 
                                    normalized_distance: float) -> float:
        """
        Calcula puntuación para evaluar la calidad de un contorno interno.
        """
        WEIGHT_AREA = 0.3
        WEIGHT_CIRCULARITY = 0.4 
        WEIGHT_CENTERING = 0.3
        
        # Puntuación por área
        if area_ratio < 0.05:
            area_score = 0.1
        elif area_ratio < 0.1:
            area_score = 0.4
        elif area_ratio < 0.2:
            area_score = 0.8
        elif area_ratio < 0.4:
            area_score = 0.6
        else:
            area_score = 0.3
        
        # Puntuación por circularidad
        if circularity > 0.9:
            circularity_score = 1.0
        elif circularity > 0.7:
            circularity_score = 0.8
        elif circularity > 0.5:
            circularity_score = 0.6
        elif circularity > 0.3:
            circularity_score = 0.4
        else:
            circularity_score = 0.1
        
        # Puntuación por centrado
        if normalized_distance < 0.05:
            centering_score = 1.0
        elif normalized_distance < 0.1:
            centering_score = 0.8
        elif normalized_distance < 0.2:
            centering_score = 0.5
        elif normalized_distance < 0.3:
            centering_score = 0.2
        else:
            centering_score = 0.0
        
        total_score = (
            area_score * WEIGHT_AREA +
            circularity_score * WEIGHT_CIRCULARITY + 
            centering_score * WEIGHT_CENTERING
        )
        
        return total_score