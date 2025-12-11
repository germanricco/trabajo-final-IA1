import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

class Segmentator:
    """
    Responsabilidad: Localizar objetos en la imagen binaria y extraer sus mascaras.
    Aplica soldadura morfologica adaptativa para corregir objetos rotos.
    """
    def __init__ (self,
                  min_area: float = 50.0,
                  merge_distance: int = 25):
        """
        Inicializa el segmentador.

        Args:
            * min_area (float): Área mínima absoluta para considerar un contorno válido.
            * merge_distance (int): Distancia máxima para fusionar contornos.
        """

        self.min_area = min_area
        self.merge_distance = merge_distance
        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ Segmentator inicializado correctamente")

    def process(self, binary_image: np.ndarray) -> Dict:
        """
        Procesa una imagen binaria para encontrar y segmentar contornos.
        Retorna bounding boxes y mascaras procesadas.
        """

        # 1. Encontrar contornos externos.
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"bounding_boxes": [], "masks": [], "total_objects": 0}
        
        # 2. Filtrado por Area Absoluta y Aspect Ratio extremo
        raw_contours = []
        raw_bboxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filtro 1: Ruido muy pequeño
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filtro 2: Relacion de aspecto extrema
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio > 25 or aspect_ratio < 0.05:
                continue

            raw_contours.append(cnt)
            raw_bboxes.append((x, y, w, h))

        # 3. Fusion de partes rotas
        merged_bboxes = self._merge_broken_parts(raw_bboxes)

        # 4. Extraer mascaras (soldadura + limpieza)
        masks = self._extract_masks_from_bboxes(binary_image, merged_bboxes)

        return {
            "bounding_boxes": [tuple(b) for b in merged_bboxes],
            "masks": masks,
            "total_objects": len(masks)
        }


    def _merge_broken_parts(self, bboxes: List[List[int]]) -> List[List[int]]:
        """
        Algoritmo iterativo para fusionar rectángulos cercanos.
        """
        if not bboxes: return []
        
        # Convertir a formato [x1, y1, x2, y2] para facilitar matemáticas
        rects = []
        for (x, y, w, h) in bboxes:
            rects.append([x, y, x + w, y + h])
            
        while True:
            merged = False
            new_rects = []
            used = [False] * len(rects)
            
            for i in range(len(rects)):
                if used[i]: continue
                
                # Rectángulo base
                r1 = rects[i]
                current_merge = r1
                
                for j in range(i + 1, len(rects)):
                    if used[j]: continue
                    r2 = rects[j]
                    
                    # Verificar cercanía
                    if self._are_close(current_merge, r2, self.merge_distance):
                        # Fusionar
                        x1 = min(current_merge[0], r2[0])
                        y1 = min(current_merge[1], r2[1])
                        x2 = max(current_merge[2], r2[2])
                        y2 = max(current_merge[3], r2[3])
                        current_merge = [x1, y1, x2, y2]
                        used[j] = True
                        merged = True
                
                new_rects.append(current_merge)
            
            rects = new_rects
            if not merged: # Si no hubo cambios en esta pasada, terminamos
                break
                
        # Volver al formato (x, y, w, h)
        final_bboxes = []
        for (x1, y1, x2, y2) in rects:
            final_bboxes.append([x1, y1, x2 - x1, y2 - y1])
            
        return final_bboxes

    def _are_close(self, r1, r2, dist):
        """Devuelve True si los rectángulos están a menos de 'dist' pixeles."""
        # r = [x1, y1, x2, y2]
        
        # Expandir r1 virtualmente por la distancia
        return (r1[0] - dist < r2[2] and r1[2] + dist > r2[0] and
                r1[1] - dist < r2[3] and r1[3] + dist > r2[1])
    
    def _extract_masks_from_bboxes(self,
                                   binary_image: np.ndarray,
                                   bounding_boxes: List[Tuple]) -> List[np.ndarray]:
        """
        Extrae máscaras individuales limpias.
        Aplica "Closing  Adaptativo" para corregir objetos rotos.
        """

        masks = []
        h_img, w_img = binary_image.shape[:2]
        
        for bbox in bounding_boxes:
            # Crear una mascara del contorno
            x, y, w, h = bbox

            # Padding (seguridad para el ROI)
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            # ROI (Region of Interest)
            roi_w, roi_h = x2 - x1, y2 - y1
            # Si el ROI es inválido (0 pixels), saltamos
            if roi_w <= 0 or roi_h <= 0: continue

            # 1. Recorte del ROI original (contiene los fragmentos rotos)
            binary_roi = binary_image[y1:y2, x1:x2]

            # 1.5 Antes de soldar realizar una limpieza previa
            clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, clean_kernel)

            # 1.75 Antes de soldar realizar distinguir entre tuercas/arandelas y clavos/tornillos
            aspect_ratio = float(w) / h if h > 0 else 1.0
            is_compact = 0.6 < aspect_ratio < 1.6
            if is_compact:
                # Estrategia Conservadora. No soldar
                object_roi = cleaned_roi
            else:
                # Estrategia Agresiva. Arreglar Tornillos
                min_dim = min(w, h)
                k_size = int(min_dim * 0.20)    # kernel_size = 20% de la menor dimensión
                # Aseguramos que sea impar
                if k_size % 2 == 0: k_size += 1
                k_size = max(5, k_size) # Mínimo 5px
                # 2. Soldadura
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                # Aplicamos Closing sobre los datos crudos del ROI
                object_roi = cv2.morphologyEx(cleaned_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            
            
            # 3. Limpieza de ruido externo
            object_roi = cv2.morphologyEx(object_roi, cv2.MORPH_OPEN, clean_kernel)
            
            # 4. Seleccionar el objeto más grande del ROI
            cnts_roi, _ = cv2.findContours(object_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_roi:
                main_cnt = max(cnts_roi, key=cv2.contourArea)

                solid_mold = np.zeros_like(object_roi)
                cv2.drawContours(solid_mold, [main_cnt], -1, 255, -1)

                # Solo permitimos pixels dentro del bbox original
                rel_x = x - x1
                rel_y = y - y1

                # Hacemos AND con un rectángulo que representa el bbox original estricto
                bbox_mask = np.zeros_like(object_roi)
                cv2.rectangle(bbox_mask, (rel_x, rel_y), (rel_x + w, rel_y + h), 255, -1)
                
                # C. COMBINACIÓN FINAL (LA CORRECCIÓN)
                # Tomamos la imagen procesada (object_roi) que TIENE EL AGUJERO.
                # Usamos solid_mold para borrar el ruido externo (islas flotantes).
                # Usamos bbox_mask para cortar picos que se salgan del cuadro.
                
                # Paso 1: Filtrar ruido externo (quedarse solo con lo que está dentro del contorno principal)
                # Al hacer AND con solid_mold, borramos lo de afuera, pero mantenemos lo de adentro.
                # PERO object_roi tiene el agujero negro (0), así que 0 AND 255 = 0. ¡El agujero sobrevive!
                filtered_roi = cv2.bitwise_and(object_roi, solid_mold)
                
                # Paso 2: Recortar al BBox
                final_object_roi = cv2.bitwise_and(filtered_roi, bbox_mask)

            else:
                final_object_roi = np.zeros_like(object_roi)
            
            # 5. Reconstrucción
            full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = final_object_roi
            masks.append(full_mask)

        return masks

