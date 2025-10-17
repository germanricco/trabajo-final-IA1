import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class Segmentator:
    def __init__ (self,
                  min_contour_area: int = 10,
                  min_mask_area: int = 100,
                  merge_close_boxes: bool = True,
                  overlap_threshold: float = 0.3,
                  max_distance: int = 20):
        """
        Inicializa el segmentador con parámetros específicos.

        Args:
            min_contour_area (int): Área mínima para considerar un contorno válido.
            merge_close_boxes (bool): Si es True, fusiona cajas delimitadoras cercanas.
            overlap_threshold (float): Umbral de solapamiento para fusionar cajas. (entre 0.2 y 0.4)
            max_distance (int): Distancia máxima entre cajas para considerarlas cercanas. (10
        """

        self.min_contour_area = min_contour_area
        self.min_mask_area = min_mask_area
        self.merge_close_boxes = merge_close_boxes
        self.overlap_threshold = overlap_threshold
        self.max_distance = max_distance

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
        
        # 5. Extraer mascaras individuales para cada contorno
        masks = self._extract_masks(binary_image, valid_contours)
        print(f"    Mascaras extraidas: {len(masks)}")

        # 6. Filtrar objetos por area minima
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
                print(f"Contorno removido por puntos: {len(contour)}")
                continue
            
            # Filtrar por area minima
            area = cv2.contourArea(contour)
            if area <= self.min_contour_area:
                print(f"Contorno removido por area: {area}")
                continue

            filtered_contours.append(contour)
        
        return filtered_contours
    
    def _extract_bounding_boxes(self, contours: List[np.ndarray]) -> List[Tuple]:
        """
        Extrae cajas delimitadoras rectangulares (axis-aligned) para cada contorno.

        Args:
            contours (List[np.ndarray]): Lista de contornos válidos.

        Returns:
            List[Tuple]: Lista de cajas delimitadoras (x, y, w, h).
        """

        bounding_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        return bounding_boxes
    
    # def _merge_bounding_boxes(self,
    #                           bounding_boxes: List[Tuple],
    #                           contours: List[np.ndarray]) -> Tuple[List, List]:
    #     """
    #     Fusiona cajas delimitadoras cercanas o que se sobrelapen.

    #     # Aplica distintas estrategias
    #     1. Una caja contiene a la otra
    #     2. Hay solapamiento
    #     3. Las cajas estan cercanas entre si

    #     Args:
    #         bounding_boxes (List): Lista de cajas delimitadoras como (x, y, w, h).
    #         contours (List): Lista de contornos que corresponden a cada caja.

    #     Returns:
    #         tuple: (List de cajas fusionadas, List de contornos correspondientes)
    #     """

    #     # Si solo hay una caja, no hay nada que fusionar
    #     if len(bounding_boxes) <= 1:
    #         return bounding_boxes, contours

    #     initial_count = len(bounding_boxes)
    #     print(f"Comenzando proceso de fusion de {initial_count} bboxes")
        
    #     # Algoritmo de fusión iterativo
    #     merged_boxes = bounding_boxes.copy()
    #     merged_contours = contours.copy()
    #     changed = True
    #     #!Debugging
    #     iterations = 0

    #     # Iterativamente fusionar hasta que no haya más cambios
    #     while changed and len(merged_boxes) > 1:
    #         changed = False
    #         new_boxes = []
    #         new_contours = []
    #         used = [False] * len(merged_boxes)
    #         merges_this_round = 0

    #         for i in range(len(merged_boxes)):
    #             if used[i]:
    #                 continue

    #             current_box = list(merged_boxes[i])
    #             current_contours = [merged_contours[i]]

    #             # Compara con las demás cajas aun no utilizadas
    #             for j in range(i + 1, len(merged_boxes)):
    #                 if used[j]:
    #                     continue
                    
    #                 # Verifica si deben fusionarse
    #                 if self._should_merge(current_box, merged_boxes[j]):
    #                     # Fusionar las cajas
    #                     current_box = list(self._merge_two_boxes(current_box, merged_boxes[j]))
    #                     current_contours.append(merged_contours[j])
    #                     used[j] = True
    #                     changed = True
    #                     #!Debugging
    #                     merges_this_round += 1

    #             new_boxes.append(tuple(current_box))

    #             # Para contornos fusionados, mantener el más grande
    #             largest_contour = max(current_contours, key=cv2.contourArea)
    #             new_contours.append(largest_contour)
    #             used[i] = True

    #         merged_boxes = new_boxes
    #         merged_contours = new_contours

    #         iterations += 1
    #         print(f"Iteracion {iterations}: {merges_this_round} fusiones realizadas, {len(merged_boxes)} cajas restantes")
            
    #         # Safety: don't run forever
    #         if iterations > 10:
    #             print("  Warning: Reached maximum iterations")
    #             break

    #     final_count = len(merged_boxes)
    #     print(f"Merge complete: {initial_count} → {final_count} boxes ({initial_count - final_count} merges)")

    #     return merged_boxes, merged_contours

    # def _should_merge(self, box1, box2):
    #     """
    #     Determina si dos cajas deben fusionarse basandose en distintos criterios.
    #     """
    #     # Estrategia 1. Si una caja contiene a la otra
    #     if self._is_bbox_contained(box1, box2):
    #         return True
            
    #     # Estrategia 2. Si hay solapamiento
    #     overlap_ratio = self._calculate_bbox_overlap_ratio(box1, box2)
    #     if overlap_ratio > self.overlap_threshold:
    #         return True
            
    #     # Estrategia 3. Si son cercanas
    #     if self._are_boxes_proximate(box1, box2):
    #         return True
            
    #     return False

    # def _is_bbox_contained(self, box1: Tuple, box2: Tuple) -> bool:
    #     """
    #     Verifica si una caja esta completamente contenida en otra.
    #     Capura si pequeños contornos estan dentro de la pieza principal

    #     Args:
    #         box1 (Tuple): (x, y, w, h) cords y dimensiones
    #         box2 (Tuple): (x, y, w, h) cords y dimensiones

    #     Returns:
    #         bool: True si una caja esta completamente contenida en otra, False de lo contrario.
    #     """
    #     x1, y1, w1, h1 = box1
    #     x2, y2, w2, h2 = box2

    #     # Verificar si caja 1 esta completamente contenida en caja 2
    #     contained_in_2 = (x1 >= x2 and y1 >= y2 and 
    #                  x1 + w1 <= x2 + w2 and 
    #                  y1 + h1 <= y2 + h2)
    
    #     # Verificar si caja 2 esta completamente contenida en caja 1
    #     contained_in_1 = (x2 >= x1 and y2 >= y1 and 
    #                     x2 + w2 <= x1 + w1 and 
    #                     y2 + h2 <= y1 + h1)
        
    #     return contained_in_1 or contained_in_2
    
    # def _calculate_bbox_overlap_ratio(self, box1: Tuple, box2: Tuple) -> float:
    #     """
    #     Calcula el ratio de solapamiento entre dos cajas basados en el area
    #     de la caja mas pequena. (mas sensible a los pequenos contornos)

    #     Args:
    #         box1 (Tuple): (x, y, w, h) cords y dimensiones
    #         box2 (Tuple): (x, y, w, h) cords y dimensiones

    #     Returns:
    #         float: Ratio de solapamiento (0.0 a 1.0) relarivo a la caja mas pequena
    #     """
    #     x1, y1, w1, h1 = box1
    #     x2, y2, w2, h2 = box2

    #     # Calcula las coordenadas de la intersección
    #     inter_x1 = max(x1, x2)
    #     inter_y1 = max(y1, y2)
    #     inter_x2 = min(x1 + w1, x2 + w2)
    #     inter_y2 = min(y1 + h1, y2 + h2)

    #     # Verificar si no hay interseccion
    #     if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
    #         return 0.0
        
    #     intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    #     area1 = w1 * h1
    #     area2 = w2 * h2

    #     # usa el area menor como denominador
    #     smaller_area = min(area1, area2)
    #     return intersection_area / smaller_area if smaller_area > 0 else 0.0
            
    # def _are_boxes_proximate(self, box1: Tuple, box2: Tuple) -> bool:
    #     """
    #     Verifica si dos cajas estan suficientemente proximas para ser consideradas partes
    #     del mismo objeto.
    #     Usa umbral de distancia dinamico

    #     Args:
    #         box1 (Tuple): (x, y, w, h) cords y dimensiones
    #         box2 (Tuple): (x, y, w, h) cords y dimensiones

    #     Returns:
    #         bool: True si las cajas estan proximas, False de lo contrario
    #     """
    #     x1, y1, w1, h1 = box1
    #     x2, y2, w2, h2 = box2
        
    #     # Calculate the actual gap between boxes (edge-to-edge distance)
    #     horizontal_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    #     vertical_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        
    #     # If boxes overlap in either direction, gap is 0 (they're already overlapping)
    #     if horizontal_gap == 0 and vertical_gap == 0:
    #         return True  # They overlap, so definitely proximate
        
    #     # Calculate the diagonal gap (direct distance between closest edges)
    #     gap_distance = np.sqrt(horizontal_gap**2 + vertical_gap**2)
        
    #     # Dynamic threshold: larger boxes can have larger gaps between them
    #     avg_size = (max(w1, h1) + max(w2, h2)) / 2
    #     dynamic_threshold = min(self.max_distance, avg_size * 0.3)  # Up to 30% of average size
        
    #     return gap_distance < dynamic_threshold  

    # def _merge_two_boxes(self, box1: Tuple, box2: Tuple) -> Tuple:
    #     """
    #     Funcion auxiliar para unir 2 cajas
    #     """
    #     x1, y1, w1, h1 = box1
    #     x2, y2, w2, h2 = box2
        
    #     new_x = min(x1, x2)
    #     new_y = min(y1, y2)
    #     new_w = max(x1 + w1, x2 + w2) - new_x
    #     new_h = max(y1 + h1, y2 + h2) - new_y
        
    #     return (new_x, new_y, new_w, new_h)
    

    # def _extract_masks(self, binary_image: np.ndarray, 
    #                   contours: List[np.ndarray]) -> List[np.ndarray]:
    #     """
    #     Extrae máscaras binarias individuales para cada objeto segmentado.
        
    #     Args:
    #         binary_image: Imagen binaria original
    #         contours: Lista de contornos válidos
            
    #     Returns:
    #         Lista de máscaras binarias, una por objeto
    #     """
    #     masks = []
    #     for contour in contours:
    #         # Crear máscara en blanco
    #         mask = np.zeros_like(binary_image)
    #         # Dibujar el contorno relleno
    #         cv2.drawContours(mask, [contour], -1, 255, -1)
    #         masks.append(mask)
    #     return masks
    
    
    # def _filter_by_mask_area(self, contours: List[np.ndarray],
    #                     bounding_boxes: List[Tuple],
    #                     masks: List[np.ndarray]) -> Dict:
    #     """
    #     Filtra los contornos, bounding boxes y mascaras por area minima.

    #     Args:
    #         contours: Lista de contornos
    #         bounding_boxes: Lista de bounding boxes
    #         masks: Lista de mascaras

    #     Returns:
    #         Diccionario con los elementos filtrados
    #     """
    #     valid_contours = []
    #     valid_bboxes = []
    #     valid_masks = []
    #     removed_count = 0

    #     for i, (cnt, bbox, mask) in enumerate(zip(contours, bounding_boxes, masks)):
    #         # Calcular el area de la mascara (num de pixeles blancos)
    #         mask_area = np.sum(mask>0)

    #         if mask_area >= self.min_contour_area:
    #             valid_contours.append(cnt)
    #             valid_bboxes.append(bbox)
    #             valid_masks.append(mask)
    #         else:
    #             removed_count += 1
    #             print(f"Contorno {i} removido por area de mascara: {mask_area}")

    #     if removed_count > 0:
    #         print(f"Contornos removidos: {removed_count}")

    #     return {
    #         'contours': valid_contours,
    #         'bounding_boxes': valid_bboxes,
    #         'masks': valid_masks,
    #     }
    
    # def _calculate_statistics(self, contours: List[np.ndarray], 
    #                         bounding_boxes: List[Tuple]) -> Dict:
    #     """
    #     Calcula estadísticas básicas sobre la segmentación.
        
    #     Args:
    #         contours: Lista de contornos
    #         bounding_boxes: Lista de bounding boxes
            
    #     Returns:
    #         Diccionario con estadísticas
    #     """
    #     if not contours:
    #         return {
    #             'total_objects': 0,
    #             'average_area': 0,
    #             'total_area': 0,
    #             'average_aspect_ratio': 0
    #         }
        
    #     areas = [cv2.contourArea(cnt) for cnt in contours]
    #     bbox_areas = [w * h for (_, _, w, h) in bounding_boxes]
    #     aspect_ratios = [max(w, h) / min(w, h) for (_, _, w, h) in bounding_boxes]
        
    #     return {
    #         'total_objects': len(contours),
    #         'average_area': np.mean(areas),
    #         'total_area': np.sum(areas),
    #         'median_area': np.median(areas),
    #         'std_area': np.std(areas),
    #         'average_bbox_area': np.mean(bbox_areas),
    #         'average_aspect_ratio': np.mean(aspect_ratios),
    #         'coverage_ratio': np.mean(areas) / np.mean(bbox_areas) if np.mean(bbox_areas) > 0 else 0
    #     }

    def _merge_bounding_boxes(self, 
                            bounding_boxes: List[Tuple], 
                            contours: List[np.ndarray]) -> Tuple[List, List]:
        """
        Fusiona bounding boxes según criterios de superposición y proximidad
        """
        if len(bounding_boxes) <= 1:
            return bounding_boxes, contours
        
        # Convertir a formato (x1, y1, x2, y2) para easier calculations
        boxes = [(x, y, x + w, y + h) for x, y, w, h in bounding_boxes]
        
        # Inicializar grupos
        groups = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
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
                print(f"      ❌ Descartado objeto {i+1}: área de máscara insuficiente ({mask_area} píxeles)")
        
        return {
            "contours": valid_contours,
            "bounding_boxes": valid_bboxes,
            "masks": valid_masks,
            "removed_count": removed_count
        }

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
