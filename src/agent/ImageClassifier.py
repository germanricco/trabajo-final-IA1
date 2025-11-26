import os
import logging
import random
import pickle
from collections import Counter
import matplotlib.image as mpimg

# Componentes del pipeline
from src.agent.ImagePreprocessor import ImagePreprocessor
from src.agent.Segmentator import Segmentator
from src.agent.ContourManager import ContourManager
from src.agent.FeatureExtractor import FeatureExtractor
from src.agent.DataPreprocessor import DataPreprocessor
from src.agent.KMeans import KMeansModel
from src.agent.Visualization import ClusterVisualizer

class ImageClassifier:
    """
    Subsistema de Vision: Encargado de cargar datos, entrenar KMeans
    validar resultados y clasificar nuevas imagenes.
    """

    def __init__(self, models_dir="models"):
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir

        # Componentes del pipeline
        self.img_prep = ImagePreprocessor(
            target_size = (800,600),
            blur_kernel_size = (7, 7),
            binarization_block_size = 11,
            binarization_C = 3,
            open_kernel_size = (5, 5),
            close_kernel_size = (9, 9)
        )

        self.segmentator = Segmentator(
            min_contour_area=100,
            min_mask_area=200,
            merge_close_boxes=True,
            overlap_threshold=0.2,
            max_distance=5
        )

        self.feature_extractor = FeatureExtractor()
        self.data_prep = DataPreprocessor()
        self.model = KMeansModel(n_clusters=4, n_init=10)

        self.cluster_mapping = {}  # Mapear ID de cluster a etiqueta semántica
        self.is_ready = False

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
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
    

    def _pipeline_process(self, image_paths_with_labels):
        """
        Ejecuta Preproceso -> Segmentación -> Feature Extraction para una lista de imagenes.
        Retorna:
            features (lista de dicts), labels (lista de strs), bboxes (lista)
        """
        features_list = []
        labels_list = []
        
        for img_path, label in image_paths_with_labels:
            try:
                # 1. Pipeline visual
                image = mpimg.imread(img_path)

                processed_img = self.img_prep.process(image)
                seg_res = self.segmentator.process(processed_img)
                
                if seg_res['total_objects'] == 0:
                    continue

                # 2. Extracción
                feats = self.feature_extractor.extract_features(seg_res['bounding_boxes'], seg_res['masks'])
                
                # 3. Acumulación (Asumimos que si la foto es de "tornillos", todos los objetos lo son)
                features_list.extend(feats)
                if label:
                    labels_list.extend([label] * len(feats))
                    
            except Exception as e:
                self.logger.warning(f"Error procesando {img_path}: {e}")
                
        return features_list, labels_list

    def train(self, data_path):
        """
        Entrena el modelo de K-Means.
        """
        # 1. División de datos
        train_set, val_set = self.load_and_split_data(data_path, split_ratio=0.8)
        
        # 2. Procesamiento del SET DE ENTRENAMIENTO
        print("⚙️ Procesando set de entrenamiento...")
        train_features, train_labels = self._pipeline_process(train_set)
        
        # 3. Normalización (FIT + TRANSFORM)
        target_cols = self.feature_extractor.get_recommended_features()
        X_train = self.data_prep.fit_transform(train_features, target_features=target_cols)
        
        # 4. Entrenamiento K-Means
        print("🧠 Entrenando K-Means...")
        self.model.fit(X_train)
        
        # 5. Mapeo Cluster -> Etiqueta (Usando las etiquetas reales de Train)
        predicted_clusters = self.model.predict(X_train)
        self._create_cluster_mapping(predicted_clusters, train_labels)
        
        self.is_ready = True
        
        # 6. Validación (Opcional pero recomendada aquí)
        accuracy = self.validate(val_set)
        
        self.save_model()
        return accuracy
    
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
        X_val = self.data_prep.transform(val_features)
        
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

    def predict_single_image(self, image_path):
        """Interfaz simple para predecir nuevos datos."""
        if not self.is_ready:
            raise RuntimeError("Modelo no entrenado.")
            
        # Pipeline para una sola imagen (label=None)
        features, _ = self._pipeline_process([(image_path, None)])
        
        if not features:
            return []
            
        # Normalizar y Predecir
        X = self.data_prep.transform(features)
        cluster_ids = self.model.predict(X)
        
        # Traducir IDs a nombres
        results = []
        for cid in cluster_ids:
            label = self.cluster_mapping.get(cid, "desconocido")
            results.append(label)
            
        return results
    
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
        with open(os.path.join(self.models_dir, "vision_system.pkl"), "wb") as f:
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
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    classifier = ImageClassifier()
    # Aquí podrías agregar código para entrenar o probar el clasificador

