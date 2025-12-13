import os
import logging
import random
import pickle
import cv2
from collections import Counter

# Componentes del pipeline
from src.vision.preprocessor import ImagePreprocessor
from src.vision.segmentator import Segmentator
from src.vision.features import FeatureExtractor
from src.vision.data_prep import DataPreprocessor
from src.vision.kmeans import KMeansModel

class ImageClassifier:
    """
    Subsistema de Vision: Encapsula toda la complejidad del ML (K-Means).
    """

    def __init__(self, models_dir="models"):
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir

        # Configuracion de Componentes
        self.img_prep = ImagePreprocessor(target_size = (960,1280),
                                gamma = 1.7,
                                d_bFilter = 8,
                                binarization_block_size = 27,
                                binarization_C = -4,
                                open_kernel_size = (3, 3),
                                close_kernel_size = (3, 3),
                                clear_border_margin = 5)
    
        self.segmentator = Segmentator(
            min_area = 100,
            merge_distance = 15
        )

        self.feature_extractor = FeatureExtractor()

        self.data_prep = DataPreprocessor()

        self.model = KMeansModel(
            n_clusters=4,
            n_init=50
        )

        # Pesos definidos para cada feature
        self.feature_weights = {
            'radius_variance': 7.0,     # Apoyo para circle_ratio
            'circle_ratio': 16.0,       # Principal separador entre tuercas vs. arandelas
            'hole_confidence': 2.0,
            'aspect_ratio': 3.0,
            'solidity': 1.0,
        }

        self.cluster_mapping = {}
        self.is_ready = False

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def predict(self, image_path: str) -> list:
        """
        Realiza la inferencia completa sobre una imagen nueva.
        Retorna una lista de diccionarios con toda la info necesaria para la UI.
        """
        if not self.is_ready:
            self.logger.error("Intento de prediccion con modelo no entrenado.")
            raise RuntimeError("El modelo no está entrenado.")

        if not os.path.exists(image_path):
            self.logger.error(f"Imagen no encontrada: {image_path}")
            return []
        
        # Carga con OpenCV
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Error de lectura (cv2) en: {image_path}")
            return []
        
        # Preprocesamiento + Segmentacion
        try:
            processed_img = self.img_prep.process(image)
            seg_res = self.segmentator.process(processed_img)
        except Exception as e:
            self.logger.error(f"Fallo en pipeline de preprocesamiento/segmentacion: {e}")
            return []
        
        if seg_res['total_objects'] == 0:
            self.logger.info(f"Ningún objeto detectado en {os.path.basename(image_path)}")
            return []

        # Extracción de Features
        features_list = self.feature_extractor.extract_features(
            seg_res['bounding_boxes'],
            seg_res['masks']
        )

        # Preprocesamiento de Datos (Normalización + Pesos)
        try:
            X = self.data_prep.transform(
                features_list,
                weights=self.feature_weights
            )
        except Exception as e:
            self.logger.critical(f"DataPreprocessor no ajustado: {e}")
            return []
        
        # Inferencia (K-Means)
        cluster_ids = self.model.predict(X)
        
        # Construcción de Respuesta
        results = []
        filename = os.path.basename(image_path)
        
        self.logger.info(f"--- 🔍 INFERENCIA: {filename} ({len(cluster_ids)} objetos) ---")
        
        for i, cid in enumerate(cluster_ids):
            label = self.cluster_mapping.get(cid, "desconocido")
            raw_feats = features_list[i]
            
            # --- LOGS DE DEBUG (La "Radiografía") ---
            # Solo se verán si configuras logging.setLevel(logging.DEBUG)
            debug_msg = [f"\n   OBJETO #{i} -> Predicción: [{label.upper()}] (Cluster {cid})"]
            
            # Análisis específico según tipo
            if raw_feats.get('hole_confidence', 0) > 0.5:
                # Caso Tuerca/Arandela
                c_ratio = raw_feats.get('circle_ratio', 0)
                rad_var = raw_feats.get('radius_variance', 0)
                debug_msg.append(f"     ├─ ⭕ Circle Ratio: {c_ratio:.4f} (Arandela > 0.90 | Tuerca < 0.88)")
                debug_msg.append(f"     ├─ 📏 Radius Var:   {rad_var:.4f} (Arandela ~ 0.02 | Tuerca > 0.05)")
                
                # Alertas de ambigüedad
                if label == 'arandelas' and c_ratio < 0.90:
                    debug_msg.append("     ⚠️  SOSPECHOSO: Clasificado Arandela pero es poco circular.")
                if label == 'tuercas' and c_ratio > 0.90:
                    debug_msg.append("     ⚠️  SOSPECHOSO: Clasificado Tuerca pero es muy circular.")
            else:
                # Caso Clavo/Tornillo
                asp_ratio = raw_feats.get('aspect_ratio', 0)
                solidity = raw_feats.get('solidity', 0)
                debug_msg.append(f"     ├─ 📏 Aspect Ratio: {asp_ratio:.4f} (Clavo > 15 | Tornillo ~3-8)")
                debug_msg.append(f"     ├─ ⬛ Solidity:     {solidity:.4f}")
            
            self.logger.debug("\n".join(debug_msg))
            # ----------------------------------------


            # Recuperamos el bbox que guardamos en el diccionario de features
            bbox = features_list[i].get('bbox')
            
            results.append({
                "label": label,
                "bbox": bbox,
                "cluster_id": int(cid)
            })
        
        print(f"Resultados de clasificación para {filename}: {results}")
        return results

    def _pipeline_process(self, image_paths_with_labels):
        """
        Método interno: Orquesta Preproceso -> Segmentación -> Features.
        """
        features_list = []
        labels_list = []
        
        for img_path, label in image_paths_with_labels:
            try:
                # Carga con OpenCV para consistencia de canales (BGR)
                image = cv2.imread(img_path)
                if image is None: continue

                processed_img = self.img_prep.process(image)
                seg_res = self.segmentator.process(processed_img)
                
                if seg_res['total_objects'] == 0:
                    continue

                feats = self.feature_extractor.extract_features(seg_res['bounding_boxes'], seg_res['masks'])
                
                features_list.extend(feats)
                if label:
                    labels_list.extend([label] * len(feats))
                    
            except Exception as e:
                self.logger.warning(f"Error procesando {img_path}: {e}")
                
        return features_list, labels_list
    
    def load_and_split_data(self, data_path, split_ratio=0.8):
        """
        Carga las rutas de imagenes y las divide en entrenamiento y validación.
        Retorna dos listas de tuplas: [(ruta_imagen, etiqueta_real), ...]
        """

        train_data = []
        val_data = []
        classes = ["arandelas", "clavos", "tornillos", "tuercas"]

        for label in classes:
            dir_path = os.path.join(data_path, label)
            if not os.path.exists(dir_path):
                self.logger.warning(f"Directorio no encontrado: {dir_path}")
                continue

            # Obtenemos todas las imágenes en este directorio
            files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            full_paths = [(os.path.join(dir_path, f), label) for f in files]

            # Mezclamos y aleatoriamente
            random.shuffle(full_paths)

            # Dividimos
            split_idx = int(len(full_paths) * split_ratio)
            train_data.extend(full_paths[:split_idx])
            val_data.extend(full_paths[split_idx:])

        self.logger.info(f"Datos cargados. Entrenamiento: {len(train_data)}, Validación: {len(val_data)}.")
        return train_data, val_data

    def train(self, data_path: str, attempts: int = 15) -> float: # Subimos intentos por default
        """
        Entrena buscando un modelo que no solo tenga buena precisión,
        sino que logre distinguir las 4 clases fundamentales.
        """
        self.logger.info(f"🏆 Iniciando Torneo de Entrenamiento ({attempts} rondas)...")
        
        best_accuracy = 0.0
        current_weights = self.feature_weights
        
        # Lista de clases obligatorias
        REQUIRED_CLASSES = {"arandelas", "clavos", "tornillos", "tuercas"}

        for i in range(1, attempts + 1):
            self.logger.debug(f"--- Ronda {i}/{attempts} ---")

            # Carga y División de datos
            train_set, val_set = self.load_and_split_data(data_path, split_ratio=0.8)
            
            # Pipeline de entrenamiento Procesamiento
            train_features, train_labels = self._pipeline_process(train_set)
            if not train_features:
                self.logger.warning("Ronda {i}: No se extrajeron features de entrenamiento.")
                continue

            target_cols = self.feature_extractor.get_recommended_features()
            
            # Instancias temporales
            temp_data_prep = DataPreprocessor() 
            # n_init alto para obligar a K-Means a probar muchas semillas
            temp_model = KMeansModel(n_clusters=4, n_init=50, max_iters=500) 
            
            # Fit
            X_train = temp_data_prep.fit_transform(
                train_features,
                target_features=target_cols,
                weights=current_weights
            )
            
            temp_model.fit(X_train)
            
            # Mapeo de Clusters a Etiquetas
            predicted_clusters = temp_model.predict(X_train)
            temp_mapping = {}
            found_classes = set()
            
            for k in range(temp_model.n_clusters):
                # Indices de los datos que cayeron en el cluster k
                indices = [idx for idx, x in enumerate(predicted_clusters) if x == k]
                if indices:
                    # Etiquetas reales de esos datos
                    labels = [train_labels[idx] for idx in indices]
                    # Votacion mayoritaria
                    most_common = Counter(labels).most_common(1)[0][0]
                    temp_mapping[k] = most_common
                    found_classes.add(most_common)
                else:
                    temp_mapping[k] = "desconocido"
            
            # Filtro de Calidad: Diversidad
            missing_classes = REQUIRED_CLASSES - found_classes
            if missing_classes:
                self.logger.warning(f"Ronda {i} descartada. Clases no encontradas {missing_classes}")
                continue # Saltamos directamente al siguiente intento
            
            # Validación
            val_features, val_labels = self._pipeline_process(val_set)
            if not val_features: continue
            
            # Transformamos validacion
            X_val = temp_data_prep.transform(val_features, weights=current_weights)
            predictions = temp_model.predict(X_val)
            
            hits = 0
            errors = []

            for idx, cid in enumerate(predictions):
                pred_lbl = temp_mapping.get(cid, "desconocido")
                real_lbl = val_labels[idx]

                if pred_lbl == real_lbl:
                    hits += 1
                else:
                    # Guardamos el error para reporte
                    errors.append(f"   [Esperado: {real_lbl.upper()}] -> [Predijo: {pred_lbl.upper()}]")
            
            current_accuracy = hits / len(predictions)
            
            self.logger.info(f" Ronda {i}: Precisión {current_accuracy:.2%}")

            # Reporte de Errores
            if errors:
                self.logger.info(f"   ❌ Fallos en Validación ({len(errors)}):")
                for err in errors:
                    self.logger.info(err)
            else:
                self.logger.info("   ✅ Validación perfecta (0 errores).")
            
            # 6. Selección del Campeón
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                
                self.model = temp_model
                self.data_prep = temp_data_prep
                self.cluster_mapping = temp_mapping
                self.is_ready = True
                
                self.save_model()
                self.logger.info(f"✨ ¡Nuevo Campeón! ({best_accuracy:.2%}) Guardado.")
                
                if best_accuracy > 0.98:
                    self.logger.info("🏆 Precisión sobresaliente alcanzada, finalizando torneo.")
                    break
        
        # Restauración final
        if self.is_ready:
            self.load_model()
            self.logger.info(f"🏁 Torneo finalizado. Ganador: {best_accuracy:.2%}")
        else:
            self.logger.error("❌ Fallo total: Ningún modelo logró identificar las 4 clases.")
            
        return best_accuracy
    
    def validate(self, val_set_paths):
        """
        Usa el 20% restante para medir la precisión.
        """
        if not self.is_ready:
            return 0.0
            
        print("🛡️ Validando modelo con set de prueba...")
        val_features, val_labels = self._pipeline_process(val_set_paths)
        
        if not val_features:
            return 0.0

        # Normalización (SOLO TRANSFORM, usar medias de train)
        X_val = self.data_prep.transform(val_features, weights=self.feature_weights)
        
        # Predicción
        predictions = self.model.predict(X_val)
        
        # --- Lógica de Diagnóstico ---
        hits = 0
        errors = [] # Lista para guardar detalles de los errores
        
        # Matriz de confusión: {Etiqueta_Real: {Etiqueta_Predicha: Cantidad}}
        classes = ["arandelas", "clavos", "tornillos", "tuercas"]
        confusion_matrix = {real: {pred: 0 for pred in classes + ["desconocido"]} for real in classes}

        for i, cluster_id in enumerate(predictions):
            predicted_label = self.cluster_mapping.get(cluster_id, "desconocido")
            real_label = val_labels[i]
            
            # Actualizar matriz
            if real_label in confusion_matrix:
                confusion_matrix[real_label][predicted_label] += 1
            
            if predicted_label == real_label:
                hits += 1
            else:
                # Guardamos el error para inspección
                # Intentamos recuperar el nombre del archivo si es posible, 
                # aunque aquí val_features ya perdió el link directo con el path exacto 
                # a menos que pasemos el path en _pipeline_process. 
                # Por ahora mostramos los índices.
                errors.append(f"❌ Esperaba '{real_label.upper()}' -> Predijo '{predicted_label.upper()}' (Cluster {cluster_id})")

        accuracy = hits / len(predictions)
        
        # --- IMPRESIÓN DEL REPORTE ---
        print(f"\n📊 REPORTE DE CLASIFICACIÓN (Precisión: {accuracy:.2%})")
        print("-" * 60)
        
        # Imprimir Matriz de Confusión bonita
        header = f"{'REAL / PRED':<15} | " + " | ".join([f"{c[:4].upper():<4}" for c in classes])
        print(header)
        print("-" * 60)
        
        for real in classes:
            row_str = f"{real:<15} | "
            for pred in classes:
                count = confusion_matrix[real][pred]
                # Resaltar errores visualmente con un asterisco si count > 0 y real != pred
                marker = "*" if count > 0 and real != pred else " "
                row_str += f"{count}{marker:<3} | "
            print(row_str)
            
        print("-" * 60)
        
        if errors:
            print(f"\n⚠️  DETALLE DE ERRORES ({len(errors)} casos):")
            for err in errors[:10]: # Mostramos los primeros 10 para no saturar
                print(f"   {err}")
            if len(errors) > 10:
                print(f"   ... y {len(errors)-10} más.")
        else:
            print("\n✨ ¡CLASIFICACIÓN PERFECTA EN VALIDACIÓN!")

        return accuracy
    
    def _create_cluster_mapping(self, cluster_ids, true_labels):
        """Asigna una etiqueta mayoritaria a cada cluster."""
        self.cluster_mapping = {}
        for k in range(self.model.n_clusters):
            indices = [i for i, x in enumerate(cluster_ids) if x == k]
            if indices:
                labels = [true_labels[i] for i in indices]
                most_common = Counter(labels).most_common(1)[0][0]
                self.cluster_mapping[k] = most_common
            else:
                self.cluster_mapping[k] = "desconocido"

    def save_model(self):
        with open(os.path.join(self.models_dir, "vision_model.pkl"), "wb") as f:
            pickle.dump({
                "model": self.model,
                "mapping": self.cluster_mapping,
                "prep": self.data_prep
            }, f)
        print("💾 Sistema de visión guardado.")

    def load_model(self):
        path = os.path.join(self.models_dir, "vision_system.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.cluster_mapping = data["mapping"]
                self.data_prep = data["prep"] # Restauramos el normalizador entrenado
                self.is_ready = True
            return True
        return False