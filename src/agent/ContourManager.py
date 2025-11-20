import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

class ContourManager:
    """
    Gestiona múltiples representaciones de un contorno para la extracción óptima de características.
    """
    
    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(np.uint8)
        self.contour_external = None
        self.contour_internal = None
        self._convex_hull = None
        self._min_area_rect = None
        self._hierarchy = None

        self.logger = logging.getLogger(__name__)
        
        # Inicializar todos los contornos al crear el manager
        self.logger.debug("Inicializando contornos desde la máscara proporcionada.")
        self._initialize_contours()
    
    def _initialize_contours(self) -> None:
        """Extrae y calcula todas las representaciones del contorno de la máscara."""
        
        # Obtener el contorno original
        contours, hierarchy = cv2.findContours(
            self.mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No se encontraron contornos en la máscara.")
        
        self._contour_original = max(contours, key=cv2.contourArea)
        self._hierarchy = hierarchy[0] if hierarchy is not None else None

        # Calcular el contorno externo
        external_contours = []
        for i, contour in enumerate(contours):
            # En RETR_CCOMP, hierarchy[i][3] == -1 significa contorno externo
            if self._hierarchy is None or self._hierarchy[i][3] == -1:
                external_contours.append(contour)
        
        if not external_contours:
            raise ValueError("No se encontraron contornos externos.")
        
        self._contour_external = max(external_contours, key=cv2.contourArea)
        
        # Buscar contornos internos (hijos del contorno externo)
        self._find_internal_contours(contours)
        
        # Calcular el casco convexo
        self._convex_hull = cv2.convexHull(self._contour_original)
        
        # Calcular el rectángulo de área mínima
        self._min_area_rect = cv2.minAreaRect(self._contour_original)

    def _find_internal_contours(self, contours: List[np.ndarray]) -> None:
        """Encuentra el contorno interno más grande (agujero de arandela)"""
        if self._hierarchy is None:
            self._contour_internal = None
            return
            
        internal_contours = []
        external_idx = None
        
        # Encontrar el índice de nuestro contorno externo
        for i, contour in enumerate(contours):
            if np.array_equal(contour, self._contour_external):
                external_idx = i
                break
        
        if external_idx is not None:
            # Buscar contornos que tengan como padre nuestro contorno externo
            for i, contour in enumerate(contours):
                if (self._hierarchy[i][3] == external_idx and 
                    cv2.contourArea(contour) > 450):  #! Filtrar por área mínima
                    internal_contours.append(contour)
            
            # Tomar el contorno interno más grande (agujero principal)
            self._contour_internal = max(internal_contours, key=cv2.contourArea) if internal_contours else None
    
    # Propiedades para acceder de forma segura a los contornos
    @property
    def external_contour(self) -> np.ndarray:
        return self._contour_external
    
    @property
    def internal_contour(self) -> Optional[np.ndarray]:
        return self._contour_internal
    
    @property
    def convex_hull(self) -> np.ndarray:
        return self._convex_hull
    
    @property
    def min_area_rectangle(self) -> cv2.RotatedRect:
        return self._min_area_rect
    
    
    def has_hole(self) -> bool:
        """Determina si el objeto tiene un agujero significativo"""
        return self._contour_internal is not None
    

    def get_actual_area(self) -> float:
        """Calcula el área real restando los agujeros internos"""
        external_area = cv2.contourArea(self._contour_external)
        
        if self.has_hole():
            internal_area = cv2.contourArea(self._contour_internal)
            return external_area - internal_area
        else:
            return external_area