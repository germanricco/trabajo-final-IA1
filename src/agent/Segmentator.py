import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class Segmentator:
    def __init__ (self,
                  min_contour_area: int = 10,
                  min_mask_area: int = 100,
                  merge_close_boxes: bool = True,
                  overlap_threshold: float = 0.3,
                  max_distance: int = 20,
                  max_aspect_ratio: float = 10.0):
        """
        Inicializa el segmentador con parámetros específicos.

        Args:
            min_contour_area (int): Área mínima para considerar un contorno válido.
            merge_close_boxes (bool): Si es True, fusiona cajas delimitadoras cercanas.
            overlap_threshold (float): Umbral de solapamiento para fusionar cajas. (entre 0.2 y 0.4)
            max_distance (int): Distancia máxima entre cajas para considerarlas cercanas. (10)
            max_aspect_ratio (float): Aspecto máximo permitido para las cajas.
        """

        self.min_contour_area = min_contour_area
        self.min_mask_area = min_mask_area
        self.merge_close_boxes = merge_close_boxes
        self.overlap_threshold = overlap_threshold
        self.max_distance = max_distance
        self.max_aspect_ratio = max_aspect_ratio

    def process(self, binary_image: np.ndarray) -> Dict:
        """
        Procesa una imagen binaria para encontrar y segmentar contornos.

        Args:
            binary_image (np.ndarray): Imagen binaria de entrada.

        Returns:
            Dict: Diccionario con las cajas delimitadoras de los contornos encontrados.
        """

        print(f"🚀 INICIANDO PROCESO DE SEGMENTACIÓN")

        # 1. Encuentrar contornos en la imagen binaria
        contours = self._find_contours(binary_image)
        print(f"   Contornos encontrados: {len(contours)}")

        # 2. Filtrar y validar los contornos encontrados
        valid_contours = self._filter_contours(contours)
        print(f"   Contornos validos: {len(valid_contours)}")

        # 3. Obtener cajas delimitadoras de los contornos válidos
        bounding_boxes = self._extract_bounding_boxes(valid_contours)
        print(f"   Bounding boxes: {len(bounding_boxes)}")

        # 4. Fusionar cajas delimitadoras cercanas si esta permitido
        if self.merge_close_boxes:
            # Nota. Se actuaizan tanto las bbox como los contornos
            bounding_boxes, valid_contours = self._merge_bounding_boxes(
                bounding_boxes, valid_contours
            )
            print(f"   Bounding boxes despues de fusion: {len(bounding_boxes)}")
        
        # 4.b Filtrar por relacion de aspecto demasiado grande
        valid_contours, bounding_boxes, aspect_removed = self._filter_by_aspect_ratio(valid_contours,
                                                                      bounding_boxes)
        
        if aspect_removed:
            print(f"   Eliminados por aspecto excesivo: {aspect_removed}")

        # 5. Extraer mascaras individuales para cada contorno
        masks = self._extract_masks(binary_image, valid_contours)
        print(f"   Mascaras extraidas: {len(masks)}")

        # 6. Filtrar mascaras por area minima
        filtered_results = self._filter_by_mask_area(valid_contours, bounding_boxes, masks)

        # 7. Rellenar diccionario
        valid_contours = filtered_results["contours"]
        bounding_boxes = filtered_results["bounding_boxes"]
        masks = filtered_results["masks"]

        # 8. Calcular estadisticas de la segmentacion
        statistics = self._calculate_statistics(valid_contours, bounding_boxes, masks)

        return {
            "bounding_boxes": bounding_boxes,
            "contours": valid_contours,
            "masks": masks,
            "statistics": statistics,
            "total_objects": len(valid_contours)
        }
        
    def _find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Encuentra contornos en una imagen binaria usando el metodo RETR_EXTERNAL.

        Args:
            binary_image (np.ndarray): Imagen binaria de entrada.

        Returns:
            List[np.ndarray]: Lista de contornos encontrados donde cada uno es un array de puntos (x, y).
        """
        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_TREE,              # Para encontrar todos los contornos (RETR_EXTERNAL para contornos externos)
            cv2.CHAIN_APPROX_SIMPLE     # Comprime segmentos horizontales, verticales y diagonales
        )

        # Si no hay jerarquia, devolver contornos directamente
        if hierarchy is None:
            return contours
        
        # Filtrar contornos que no son de nivel 0 (externos)
        parent_contours = []

        #NOTA. Estrcutura de jerarquia:
        #[next, previous, first_child, parent]
        for i, contours in enumerate(contours):
            # si parent == -1 es contorno de top-level
            if hierarchy[0][i][3] == -1:
                parent_contours.append(contours)

        return parent_contours

    def _filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filtra contornos basados en el área mínima y otros criterios de calidad.

        Args:
            contours: Lista de contornos encontrados.

        Returns:
            List[np.ndarray]: Lista de contornos válidos.
        """
        
        filtered_contours = []

        for contour in contours:
            # Verificar longitud del contorno
            if len(contour) < 5:
                continue
            
            # Filtrar por area minima
            area = cv2.contourArea(contour)
            if area <= self.min_contour_area:
                continue

            filtered_contours.append(contour)
        
        return filtered_contours
    
    def _extract_bounding_boxes(self, contours: List[np.ndarray]) -> List[Tuple]:
        """
        Extrae cajas delimitadoras rectangulares (axis-aligned) para cada contorno.
        """

        bounding_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        return bounding_boxes

    def _merge_bounding_boxes(self, 
                            bounding_boxes: List[Tuple], 
                            contours: List[np.ndarray]) -> Tuple[List, List]:
        """
        Fusiona bounding boxes según criterios de superposición y proximidad
        """

        # Si solo tengo una caja, no hacer nada
        if len(bounding_boxes) <= 1:
            return bounding_boxes, contours
        
        # Convertir a formato (x1, y1, x2, y2)
        boxes = [(x, y, x + w, y + h) for x, y, w, h in bounding_boxes]
        n = len(boxes)

        # Inicializar grupos
        groups = []
        used = [False] * len(boxes)
        
        for i in range(n):
            if used[i]:
                continue
                
            current_group = [i]
            used[i] = True
            
            # Buscar boxes que se solapen o estén cerca
            changed = True
            while changed:
                changed = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue
                    
                    # Verificar si debe unirse al grupo
                    if self._should_merge_boxes(current_group, j, boxes, contours):
                        current_group.append(j)
                        used[j] = True
                        changed = True
            
            groups.append(current_group)
        
        # Fusionar boxes en cada grupo
        merged_boxes = []
        merged_contours = []
        
        for group in groups:
            if len(group) == 1:
                # No hay fusión necesaria
                idx = group[0]
                merged_boxes.append(bounding_boxes[idx])
                merged_contours.append(contours[idx])
            else:
                # Fusionar múltiples boxes
                merged_box = self._merge_box_group([bounding_boxes[i] for i in group])
                merged_contour = self._merge_contour_group([contours[i] for i in group])
                
                merged_boxes.append(merged_box)
                merged_contours.append(merged_contour)
        
        return merged_boxes, merged_contours

    def _should_merge_boxes(self, 
                          current_group: List[int], 
                          candidate_idx: int,
                          boxes: List[Tuple], 
                          contours: List[np.ndarray]) -> bool:
        """
        Determina si un bounding box debe unirse al grupo actual
        """
        candidate_box = boxes[candidate_idx]
        
        for group_idx in current_group:
            group_box = boxes[group_idx]
            
            # 1. Verificar si uno está dentro de otro
            if self._is_inside(candidate_box, group_box) or self._is_inside(group_box, candidate_box):
                return True
            
            # 2. Verificar superposición
            overlap = self._calculate_overlap(candidate_box, group_box)
            if overlap > self.overlap_threshold:
                return True
            
            # 3. Verificar proximidad
            if self._calculate_distance(candidate_box, group_box) < self.max_distance:
                return True
        
        return False

    def _is_inside(self, box1: Tuple, box2: Tuple) -> bool:
        """Verifica si box1 está completamente dentro de box2"""
        x11, y11, x21, y21 = box1
        x12, y12, x22, y22 = box2
        
        return (x11 >= x12 and y11 >= y12 and x21 <= x22 and y21 <= y22)

    def _calculate_overlap(self, box1: Tuple, box2: Tuple) -> float:
        """Calcula el porcentaje de superposición entre dos boxes"""
        x11, y11, x21, y21 = box1
        x12, y12, x22, y22 = box2
        
        # Calcular intersección
        x_left = max(x11, x12)
        y_top = max(y11, y12)
        x_right = min(x21, x22)
        y_bottom = min(y21, y22)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x21 - x11) * (y21 - y11)
        area2 = (x22 - x12) * (y22 - y12)
        
        # Retornar el máximo de las dos superposiciones posibles
        overlap1 = intersection_area / area1 if area1 > 0 else 0
        overlap2 = intersection_area / area2 if area2 > 0 else 0
        
        return max(overlap1, overlap2)

    def _calculate_distance(self, box1: Tuple, box2: Tuple) -> float:
        """Calcula la distancia mínima entre dos bounding boxes"""
        x11, y11, x21, y21 = box1
        x12, y12, x22, y22 = box2
        
        # Si se solapan, distancia es 0
        if not (x21 < x12 or x22 < x11 or y21 < y12 or y22 < y11):
            return 0.0
        
        # Calcular distancias en X e Y
        dx = max(x12 - x21, x11 - x22, 0)
        dy = max(y12 - y21, y11 - y22, 0)
        
        return np.sqrt(dx**2 + dy**2)

    def _merge_box_group(self, boxes: List[Tuple]) -> Tuple:
        """Fusiona múltiples bounding boxes en uno"""
        if not boxes:
            return (0, 0, 0, 0)
        
        # Encontrar los límites extremos
        x_min = min(box[0] for box in boxes)
        y_min = min(box[1] for box in boxes)
        x_max = max(box[0] + box[2] for box in boxes)
        y_max = max(box[1] + box[3] for box in boxes)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _merge_contour_group(self, contours: List[np.ndarray]) -> np.ndarray:
        """Fusiona múltiples contornos en uno"""
        if not contours:
            return np.array([])
        
        # Simplemente tomar el contorno más grande por ahora
        # Podrías implementar una fusión más sofisticada si es necesario
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    def _extract_masks(self, binary_image: np.ndarray,
                      contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extrae máscaras individuales limpias para cada contorno
        """
        masks = []
        
        for contour in contours:
            # Crear máscara del contorno
            contour_mask = np.zeros_like(binary_image)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            
            # Intersecar con la imagen binaria para eliminar ruido interno
            clean_mask = cv2.bitwise_and(contour_mask, binary_image)
            
            # Limpieza morfológica
            kernel = np.ones((3, 3), np.uint8)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
            
            masks.append(clean_mask)
        
        return masks

    def _filter_by_mask_area(self, 
                           contours: List[np.ndarray],
                           bounding_boxes: List[Tuple], 
                           masks: List[np.ndarray]) -> Dict:
        """
        Filtra objetos basándose en el área de píxeles blancos de la máscara
        """
        valid_contours = []
        valid_bboxes = []
        valid_masks = []
        removed_count = 0
        
        for i, (contour, bbox, mask) in enumerate(zip(contours, bounding_boxes, masks)):
            mask_area = np.sum(mask > 0)
            
            if mask_area >= self.min_mask_area:
                valid_contours.append(contour)
                valid_bboxes.append(bbox)
                valid_masks.append(mask)
            else:
                removed_count += 1
                print(f"   Objeto {i} rechazado por area de mascara")
        
        return {
            "contours": valid_contours,
            "bounding_boxes": valid_bboxes,
            "masks": valid_masks,
            "removed_count": removed_count
        }

    def _filter_by_aspect_ratio(self,
                                contours: List[np.ndarray],
                                bounding_boxes: List[Tuple]) -> Tuple[List[np.ndarray], List[Tuple], int]:
        """
        Elimina contornos y bounding boxes cuya relación de aspecto (ancho/alto o alto/ancho)
        es mayor que self.max_aspect_ratio.

        Retorna: (contours_filtered, bboxes_filtered, removed_count)
        """
        if not contours or not bounding_boxes:
            return contours, bounding_boxes, 0

        filtered_contours: List[np.ndarray] = []
        filtered_bboxes: List[Tuple] = []
        removed = 0

        for cnt, bbox in zip(contours, bounding_boxes):
            x, y, w, h = bbox
            # Evitar división por cero y cajas degeneradas
            if w <= 0 or h <= 0:
                removed += 1
                continue

            ratio = (w / h) if w >= h else (h / w)
            if ratio <= self.max_aspect_ratio:
                filtered_contours.append(cnt)
                filtered_bboxes.append(bbox)
            else:
                removed += 1

        return filtered_contours, filtered_bboxes, removed

    def _calculate_statistics(self, 
                            contours: List[np.ndarray],
                            bounding_boxes: List[Tuple],
                            masks: List[np.ndarray]) -> Dict:
        """Calcula estadísticas de la segmentación incluyendo áreas de máscaras"""
        if not contours:
            return {}
        
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        mask_areas = [np.sum(mask > 0) for mask in masks]
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
        
        return {
            "total_objects": len(contours),
            "total_contour_area": sum(contour_areas),
            "total_mask_area": sum(mask_areas),
            "average_contour_area": sum(contour_areas) / len(contour_areas),
            "average_mask_area": sum(mask_areas) / len(mask_areas),
            "max_contour_area": max(contour_areas),
            "min_contour_area": min(contour_areas),
            "max_mask_area": max(mask_areas),
            "min_mask_area": min(mask_areas),
            "average_perimeter": sum(perimeters) / len(perimeters)
        }
