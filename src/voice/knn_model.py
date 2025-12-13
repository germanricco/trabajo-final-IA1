import numpy as np
import collections
import pickle
import os

class KNNModel:
    """
    Implementación del algoritmo K-Nearest Neighbors para clasificación de audio.
    """
    def __init__(self, k=5):
        self.k = k
        self.features = None
        self.labels = None
        self.n_examples = 0
        self.is_trained = False

    def fit(self, features, labels):
        """
        Carga los datos de entrenamiento en memoria.

        Args:
            * features (list of np.array): Lista de vectores de características.
            * labels (list of str): Lista de etiquetas correspondientes.
        """
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.n_examples = self.features.shape[0]
        self.is_trained = True

    def predict(self, input_feature):
        """
        Clasifica una nueva muestra de audio comparándola con la base de conocimientos.
        
        Args:
            * input_feature (np.array): Vector de características del audio a clasificar.
            
        Returns:
            * tuple: (Etiqueta Predicha, Confianza)
                   Ej: ('contar', 0.66) si 2 de 3 vecinos dijeron 'contar'.
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado ni cargado.")

        # Cálculo vectorizado de distancias Euclidianas
        distances = np.linalg.norm(self.features - input_feature, axis=1)

        # Obtener los índices de los K vecinos más cercanos (menor distancia)
        k_indices = np.argsort(distances)[:self.k]
        
        # Recuperar las etiquetas de esos vecinos
        k_nearest_labels = self.labels[k_indices]
        
        # Votación por mayoría
        most_common = collections.Counter(k_nearest_labels).most_common(1)
        
        predicted_label = most_common[0][0]
        votes = most_common[0][1]
        
        # Retornamos etiqueta y confianza (0.0 a 1.0)
        confidence = votes / self.k
        return predicted_label, confidence

    def save(self, filepath):
        """Guarda el estado completo del modelo en disco."""
        model_data = {
            "features": self.features,
            "labels": self.labels,
            "k": self.k,
            "n_examples": self.n_examples
        }
        # Aseguramos que el directorio exista
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, filepath):
        """Restaura un modelo desde disco."""
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)
            
            self.features = model_data["features"]
            self.labels = model_data["labels"]
            self.k = model_data["k"]
            self.n_examples = model_data["n_examples"]
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error cargando modelo KNN: {e}")
            return False