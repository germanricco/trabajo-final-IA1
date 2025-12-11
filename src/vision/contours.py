import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class InternalContourCandidate:
    """
    Clase Ayudante: Almacena y calcula propiedades de un posible agujero (contorno hijo).
    """
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
        Calcula propiedades relativas al contorno externo (padre).
        """
        # 1. Área y perímetro
        self.area = float(cv2.contourArea(self.contour))
        self.perimeter = float(cv2.arcLength(self.contour, True))
        
        # 2. Circularidad (4*pi*area / perim^2)
        if self.perimeter > 0:
            self.circularity = (4 * np.pi * self.area) / (self.perimeter ** 2)
        else:
            self.circularity = 0.0
        
        # 3. Centroide del candidato
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            self.centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        else:
            # Fallback al centro del bounding box si el momento falla
            x, y, w, h = cv2.boundingRect(self.contour)
            self.centroid = (x + w/2, y + h/2)
        
        # 4. Distancia normalizada al centro del contorno externo
        # Necesitamos el centroide externo para comparar
        M_ext = cv2.moments(external_contour)
        if M_ext["m00"] != 0:
            cX_ext = M_ext["m10"] / M_ext["m00"]
            cY_ext = M_ext["m01"] / M_ext["m00"]
            
            # Distancia Euclidiana
            dist_px = np.sqrt((self.centroid[0] - cX_ext)**2 + (self.centroid[1] - cY_ext)**2)
            
            # Normalizamos usando el diámetro equivalente del objeto externo
            # (Diameter of a circle with same area)
            equivalent_diameter = np.sqrt(4 * external_area / np.pi)
            
            if equivalent_diameter > 0:
                self.normalized_distance = dist_px / equivalent_diameter
            else:
                self.normalized_distance = 1.0 # Muy lejos (error)
        else:
            self.normalized_distance = 0.0 # Asumimos concéntricos si falla
        
        # 5. Relación de área (Hole Area / Object Area)
        self.area_ratio = self.area / external_area if external_area > 0 else 0.0

@dataclass
class GeometricProperties:
    """Almacena todas las propiedades geométricas calculadas una sola vez"""
    # Basicas
    area: float
    perimeter: float

    # Forma
    aspect_ratio: float
    circularity: float
    compactness: float
    rectangularity: float
    roughness: float
    solidity: float
    extent: float
    hull_area: float

    # Avanzadas
    num_vertices: float
    circle_ratio: float 
    radius_variance:float
    hole_area_ratio: float

    # Estructurales
    has_structural_hole: bool
    hole_confidence: float

    # invariantes
    hu_moments: np.ndarray

    # Debug/Visualizacion
    convex_hull: np.ndarray
    min_area_rect: Tuple
    external_contour: np.ndarray
    internal_contour: Optional[np.ndarray] = None
    

class ContourManager:
    """
    Analiza una máscara binaria para extraer métricas geométricas robustas.
    """
    
    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(np.uint8)
        self._properties: Optional[GeometricProperties] = None
        self._calculated = False
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_properties(self) -> GeometricProperties:
        """
        Calcula todas las propiedades geométricas de una vez.
        Retorna None si la mascara no es valida
        """
        # Si ya estan calculadas, retornarlas
        if self._calculated and self._properties is not None:
            return self._properties
        
        try:
            # Encontrar contornos
            external_contour, hole_candidate = self._find_contours_hierarchical()
            
            # Valida si hay al menos un contorno externo
            if external_contour is None:
                self.logger.warning("No se encontraron contornos externos en la mascara.")
                return None
            
            # Area y Perimetro de Contorno Externo
            area = float(cv2.contourArea(external_contour))
            if area == 0: return None
            perimeter = float(cv2.arcLength(external_contour, True))
            
            # Convex Hull, Solidez y Rugosidad
            convex_hull = cv2.convexHull(external_contour)
            hull_area = float(cv2.contourArea(convex_hull))
            hull_perimeter = float(cv2.arcLength(convex_hull, True))

            solidity = float(area / hull_area) if hull_area > 0 else 0.0
            roughness = float(hull_perimeter / perimeter) if perimeter > 0 else 0.0

            # Rectangulo rotado
            min_area_rect = cv2.minAreaRect(external_contour)
            (center_x, center_y), (rw, rh), angle = min_area_rect

            dim_max = max(rw, rh)
            dim_min = min(rw, rh)
            # Aspect Ratio
            aspect_ratio = float(dim_max / dim_min) if dim_min > 0 else 0.0

            rotated_area = rw * rh
            rectangularity = float(area / rotated_area) if rotated_area > 0 else 0.0

            # Extent
            x, y, w, h = cv2.boundingRect(external_contour)
            rect_area = w * h
            extent = float(area / rect_area) if rect_area > 0 else 0.0

            # Circularidad y Compacidad
            circularity = float((4 * np.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0.0
            compactness = float((perimeter ** 2) / area) if area > 0 else 0.0
            
            # Momentos de Hu 
            M = cv2.moments(external_contour)
            hu_raw = cv2.HuMoments(M).flatten()
            hu_moments = self._normalize_hu_moments(hu_raw)

            # Centroide
            if M['m00'] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, bw, bh = cv2.boundingRect(external_contour)
                cx, cy = x + bw // 2, y + bh // 2
            
            # Deteccion de Agujeros
            has_structural_hole, hole_confidence = self._classify_structural_hole(hole_candidate)
            
            # Circle Ratio
            (_, _), radius = cv2.minEnclosingCircle(external_contour)
            circle_area = np.pi * (radius ** 2)
            circle_ratio = float(area / circle_area) if circle_area > 0 else 0.0

            # Radius Variance
            pts = external_contour.reshape(-1, 2)
            centroid = np.array([cx, cy])
            # Distancia de cada punto del borde (array de N longitudes)
            distances = np.linalg.norm(pts - centroid, axis=1)
            # Estadística: Desviación Estándar / Media (Coeficiente de Variación)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            radius_variance = float(std_dist / mean_dist) if mean_dist > 0 else 0.0

            # Num Vertices (Aproximacion Poligonal)
            epsilon = 0.015 * perimeter 
            approx = cv2.approxPolyDP(external_contour, epsilon, True)
            num_vertices = float(len(approx))

            # Hole Area Ratio
            if hole_candidate:
                hole_area_ratio = hole_candidate.area_ratio
            else:
                hole_area_ratio = 0.0
            
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
                rectangularity=rectangularity,
                roughness=roughness,
                extent=extent,
                hu_moments=hu_moments,
                has_structural_hole=has_structural_hole,
                hole_confidence=hole_confidence,
                num_vertices=num_vertices,
                circle_ratio=circle_ratio,
                radius_variance=radius_variance,
                hole_area_ratio=hole_area_ratio,
                external_contour=external_contour,
                internal_contour=hole_candidate.contour if hole_candidate else None
            )
            
            self._calculated = True
            return self._properties
            
        except Exception as e:
            self.logger.error(f"Error Critico calculando propiedades: {e}")
            return None
    
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
        
        # Jerarquia: [Next, Previous, First_Child, Parent]
        # Parent == -1 -> Contorno externo
        # Contorno externo mas grande
        external_contours = []
        external_indices = []

        for i, h in enumerate(hierarchy[0]):
            if h[3] == -1: # Es externo
                external_contours.append(contours[i])
                external_indices.append(i)
        
        if not external_contours:
            return None, None
        
        # Tomamos el indice del mayor contorno externo (mayor area)
        max_idx_local = np.argmax([cv2.contourArea(c) for c in external_contours])
        external_contour = external_contours[max_idx_local]
        external_contour_index = external_indices[max_idx_local]
        external_area = cv2.contourArea(external_contour)

        
        # Buscar candidatos a agujeros (hijos del contorno externo)
        candidates = []

        # Iteramos todos los contornos para encontrar hijo de external_contour_index
        for i, h in enumerate(hierarchy[0]):
            parent_id = h[3]
            if parent_id == external_contour_index:
                # Es un agujero dentro de nuestro objeto
                cnt = contours[i]
                hole_area = cv2.contourArea(cnt)

                # Filtro de ruido para agujeros
                # Debe ser al menos 10px y no más grande que el 60% del objeto (arandelas finas)
                if hole_area < 5 or hole_area > 0.9 * external_area:
                    continue

                # Creamos el candidato
                try:
                    candidate = InternalContourCandidate(contour=cnt)
                    candidate.calculate_properties(external_contour, external_area)

                    # Puntuamos
                    candidate.score = self._calculate_internal_contour_score(
                        candidate.area_ratio,
                        candidate.circularity,
                        candidate.normalized_distance
                    )
                    candidates.append(candidate)
                except Exception as e:
                    self.logger.warning(f"Error creando candidato interno: {e}")
        
        if not candidates:
            return external_contour, None
        
        # El mejor candidato es el agujero estructural
        best_candidate = max(candidates, key=lambda x: x.score)

        MINIMUM_SCORE_THRESHOLD = 0.5
        if best_candidate.score >= MINIMUM_SCORE_THRESHOLD:
            return external_contour, best_candidate
        
        return external_contour, None
    

    def _normalize_hu_moments(self, hu_moments: np.ndarray) -> np.ndarray:
        """
        Normaliza momentos de Hu usando transformación logarítmica.
        """
        hu_log = []
        for h in hu_moments:
            if h == 0:
                hu_log.append(0.0)
            else:
                # -sgn(h) * log10(abs(h))
                hu_log.append(-1 * np.sign(h) * np.log10(np.abs(h)))
        return np.array(hu_log)[:3] # Solo los primeros 3 son robustos [hu1, hu2, hu3]

        
    def _calculate_internal_contour_score(self, area_ratio: float, circularity: float, 
                                          normalized_distance: float) -> float:
        """
        Sistema de puntuación para decidir si un hueco es un 'Agujero Estructural'.
        """
        score = 0.0
        # Premia tamaño decente
        if area_ratio > 0.05: score += 0.3
        # Premia redondez
        if circularity > 0.5: score += 0.4
        # Premia estar centrado
        if normalized_distance < 0.2: score += 0.3
        
        return score
    

    def _classify_structural_hole(self, hole_candidate: Optional[InternalContourCandidate]) -> Tuple[bool, float]:
        """
        Decisión final basada en el candidato ganador.
        """
        if hole_candidate is None:
            return False, 0.0
        
        # Sistema de confianza
        confidence_points = 0.0
        
        if hole_candidate.area_ratio > 0.05: confidence_points += 2.0
        if hole_candidate.circularity > 0.7: confidence_points += 2.0
        if hole_candidate.normalized_distance < 0.15: confidence_points += 2.0
        
        # Umbral para decir "SÍ, TIENE AGUJERO"
        has_hole = confidence_points >= 4.0
        
        # Normalizar a 0-1
        normalized_confidence = min(confidence_points / 6.0, 1.0)
        
        return has_hole, normalized_confidence