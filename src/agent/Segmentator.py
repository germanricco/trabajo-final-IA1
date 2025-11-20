import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

class Segmentator:
    def __init__ (self,
                  min_contour_area: int = 10,
                  merge_close_boxes: bool = True,
                  overlap_threshold: float = 0.3,
                  max_distance: int = 20,
                  max_aspect_ratio: float = 10.0,
                  min_mask_area: int = 50):
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
        self.merge_close_boxes = merge_close_boxes
        self.overlap_threshold = overlap_threshold
        self.max_distance = max_distance
        self.max_aspect_ratio = max_aspect_ratio
        self.min_mask_area = min_mask_area

        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ Segmentator inicializado correctamente")

    def process(self, binary_image: np.ndarray) -> Dict:
        """
        Procesa una imagen binaria para encontrar y segmentar contornos.

        Args:
            binary_image (np.ndarray): Imagen binaria de entrada.

        Returns:
            Dict: Diccionario con las cajas delimitadoras de los contornos encontrados.
        """

        # print(f"🚀 INICIANDO PROCESO DE SEGMENTACIÓN")

        # Encuentrar contornos en la imagen binaria
        contours = self._find_contours(binary_image)
        logging.debug(f"   Contornos encontrados: {len(contours)}")

        # Filtrar y validar los contornos encontrados
        valid_contours = self._filter_contours(contours)
        logging.debug(f"   Contornos válidos después del filtrado: {len(valid_contours)}")

        # Obtener cajas delimitadoras de los contornos válidos
        bounding_boxes = self._extract_bounding_boxes(valid_contours)
        logging.debug(f"   Bounding boxes: {len(bounding_boxes)}")

        # Fusionar cajas delimitadoras cercanas si esta permitido
        if self.merge_close_boxes:
            bounding_boxes, valid_contours = self._merge_bounding_boxes(
                bounding_boxes, valid_contours
            )
        logging.debug(f"   Bounding boxes despues de fusion: {len(bounding_boxes)}")
        
        # Filtrar por relacion de aspecto demasiado grande
        valid_contours, bounding_boxes, aspect_removed = self._filter_by_aspect_ratio(valid_contours,
                                                                      bounding_boxes)
        
        if aspect_removed:
           logging.debug(f"   Contornos eliminados por aspecto: {aspect_removed}")

        # Extraer mascaras individuales para cada contorno
        masks = self._extract_masks(binary_image, valid_contours)
        logging.debug(f"   Mascaras extraidas: {len(masks)}")

        # Filtrar mascaras
        filtered_results = self._comprehensive_filtering(valid_contours, bounding_boxes, masks)
        logging.debug(f"   Mascaras despues del filtrado completo: {len(filtered_results['masks'])}")

        # Rellenar diccionario
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
    
    # === BOUNDING BOXES ===

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

# === MASCARAS ===

    def _extract_masks(self, binary_image: np.ndarray,
                    contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extrae máscaras individuales limpias y completas para cada contorno.
        Combina operaciones morfológicas robustas con relleno de agujeros.
        """
        masks = []
        # Consider making the kernel size configurable
        kernel_size = 11
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for contour in contours:
            # 1. Crear una mascara del contorno
            contour_mask = np.zeros_like(binary_image)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)  # Draw filled contour

            # 2. Perform a strong MORPH_CLOSE to fill holes and smooth the shape
            # This is crucial for filling internal gaps[citation:4]
            closed_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 3. Optional but recommended: Explicit hole filling using floodFill
            filled_mask = closed_mask.copy()
            h, w = filled_mask.shape
            
            # Create a mask for floodFill (needs to be 2 pixels wider and taller)
            floodfill_mask = np.zeros((h+2, w+2), np.uint8)
            
            # Seed point: use the centroid of the contour, likely inside the object
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Ensure the seed point is within the image bounds
                if 0 <= cX < w and 0 <= cY < h:
                    cv2.floodFill(filled_mask, floodfill_mask, (cX, cY), 255)
            
            # 4. Final intersection with the original binary image to respect overall boundaries
            # This step is now less destructive as the primary filling is done.
            clean_mask = cv2.bitwise_and(filled_mask, binary_image)
            
            # 5. A final light morphological close to clean up the edges
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            
            masks.append(clean_mask)
        
        return masks

    def _comprehensive_filtering(self,
                                 contours: List[np.ndarray],
                                 bounding_boxes: List[Tuple],
                                 masks: List[np.ndarray]) -> Dict:
        """
        Filtrado Completo en multiples etapas de mascaras
        """

        # Filtro absoluto minimo (ruido extremo)
        preliminary_filtered = self._apply_absolute_filter(contours, bounding_boxes, masks)
        self.logger.debug("Filtrado preliminar: {preliminary_filtered['removed_count']} mascaras eliminadas")
        
        if len(preliminary_filtered["masks"]) <= 1:
            return preliminary_filtered
        
        # Filtro relativo basado en el area del mayor contorno
        elif len(preliminary_filtered["masks"]) == 2:
            self.logger.debug("Aplicando filtro especial para 2 mascaras")

            return self._filter_two_masks(
                preliminary_filtered["contours"],
                preliminary_filtered["bounding_boxes"],
                preliminary_filtered["masks"]
            )
            
        # Filtro relativo basado en el area media de los contornos
        else:
            self.logger.debug("Aplicando filtro por area relativa")

            return self._filter_by_relative_area(
                preliminary_filtered["contours"],
                preliminary_filtered["bounding_boxes"],
                preliminary_filtered["masks"]
            )



    def _apply_absolute_filter(self, contours, bounding_boxes, masks):
        """Filtrado absoluto inicial para ruido muy evidente"""
        valid_contours, valid_bboxes, valid_masks = [], [], []
        removed_count = 0
        
        # Recorrer todos las mascaras, calculando el area y filtrando
        for contour, bbox, mask in zip(contours, bounding_boxes, masks):
            area = np.sum(mask > 0)
            if area >= self.min_mask_area:
                # Si el area es mayor los agregamos a los validos
                valid_contours.append(contour)
                valid_bboxes.append(bbox)
                valid_masks.append(mask)
            else:
                removed_count += 1
                self.logger.debug(f"Objeto de area {area} eliminado (filtrado absoluto)")

        return {
            "contours": valid_contours,
            "bounding_boxes": valid_bboxes, 
            "masks": valid_masks,
            "removed_count": removed_count
        }
    
    def _filter_two_masks(self, contours: List[np.ndarray],
                    bounding_boxes: List[Tuple],
                    masks: List[np.ndarray]) -> Dict:
        """
        Filtro especial para cuando hay exactamente 2 máscaras
        Toma como referencia el área de la mayor
        """
        # Calcular áreas
        areas = [np.sum(mask > 0) for mask in masks]
        max_area = max(areas)
        min_area = min(areas)
        
        # Calcular relación entre áreas
        area_ratio = min_area / max_area
        
        # Si la relación es muy baja, mantener solo la más grande
        if area_ratio < 0.3:  #! Ajusta este umbral según necesites
            # Encontrar índice del objeto más grande
            larger_idx = 0 if areas[0] > areas[1] else 1
            
            valid_contours = [contours[larger_idx]]
            valid_bboxes = [bounding_boxes[larger_idx]]
            valid_masks = [masks[larger_idx]]
            removed_count = 1
            self.logger.debug(f"Objeto {1 - larger_idx} eliminado por baja relación de área ({area_ratio:.2f})")
            
        else:
            # Mantener ambos objetos (probablemente dos piezas válidas)
            valid_contours = contours
            valid_bboxes = bounding_boxes
            valid_masks = masks
            removed_count = 0
            self.logger.debug(f"Objeto {0} y {1} mantenidos (relación de área aceptable: {area_ratio:.2f})")
        
        return {
            "contours": valid_contours,
            "bounding_boxes": valid_bboxes,
            "masks": valid_masks,
            "removed_count": removed_count,
            "area_ratio": area_ratio
        }

    def _filter_by_relative_area(self, contours, bounding_boxes, masks):
        """
        Filtra objetos basándose en el área relativa usando estadísticas simples
        """
        # Calcular todas las áreas
        mask_areas = [np.sum(mask > 0) for mask in masks]
        
        if not mask_areas:
            return {"contours": [], "bounding_boxes": [], "masks": [], "removed_count": 0}
        
        # Calcular estadísticas de referencia
        median_area = np.median(mask_areas)
        mean_area = np.mean(mask_areas)
        
        # Usar la mediana como referencia (robusta a outliers)
        reference_area = median_area
        threshold_ratio = 0.10  #! Conservar objetos con al menos 10% del área de referencia
        
        valid_contours, valid_bboxes, valid_masks = [], [], []
        removed_count = 0
        
        for i, (contour, bbox, mask, area) in enumerate(zip(contours, bounding_boxes, masks, mask_areas)):
            if area >= reference_area * threshold_ratio:
                valid_contours.append(contour)
                valid_bboxes.append(bbox)
                valid_masks.append(mask)
            else:
                removed_count += 1
                print(f"   Objeto {i} rechazado: área {area} < {reference_area * threshold_ratio:.1f}")
        
        return {
            "contours": valid_contours,
            "bounding_boxes": valid_bboxes,
            "masks": valid_masks,
            "removed_count": removed_count,
            "reference_area": reference_area
        }


    # === Estadisticas ===

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
