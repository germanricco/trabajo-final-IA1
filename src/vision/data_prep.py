import numpy as np
import pickle
from typing import List, Dict

class DataPreprocessor:
    def __init__(self):
        # Almacenamos el orden de las columnas para asegurar consistencia
        self.feature_names_order_ = None
        self.means_ = None
        self.stds_ = None
        self.is_fitted = False

    def fit(self, features_list: List[Dict], target_features: List[str] = None):
        """
        Aprende las estadísticas (media, desviación estándar) de las características especificadas en target_features.
        """
        if not features_list:
            raise ValueError("Lista de características vacía.")

        # Si no nos dicen que usar, usamos todas las disponibles
        if target_features is None:
            self.feature_names_order_ = list(features_list[0].keys())
        else:
            self.feature_names_order_ = target_features
        

        X = self._to_matrix(features_list)
        
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        
        # Evitar división por cero: si std es 0, lo cambiamos a 1
        # (significa que esa característica no varía y no aporta info, pero no debe romper el código)
        self.stds_[self.stds_ == 0] = 1.0
        
        self.is_fitted = True
        return self

    def transform(self, 
                  features_list: List[Dict],
                  weights: Dict[str, float] = None) -> np.ndarray:
        """
        Normaliza nuevos datos usando las estadísticas aprendidas en fit.
        Aplica ponderacion opcional
        """
        if not self.is_fitted:
            raise RuntimeError("El preprocesador no ha sido entrenado (fit).")
            
        X_norm = self._to_matrix(features_list)
        
        if weights:
            for feature_name, weight in weights.items():
                if feature_name in self.feature_names_order_:
                    # Encontrar el índice de la columna
                    col_idx = self.feature_names_order_.index(feature_name)
                    # Multiplicar esa columna por el peso
                    X_norm[:, col_idx] *= weight
        
        return X_norm

    def fit_transform(self,
                      features_list: List[Dict],
                      target_features: List[str] = None,
                      weights: Dict[str, float] = None) -> np.ndarray:
        self.fit(features_list, target_features)
        return self.transform(features_list, weights)

    def _to_matrix(self, features_list: List[Dict]) -> np.ndarray:
        """Convierte lista de dicts a matriz numpy respetando el orden."""
        matrix = []
        for f in features_list:
            # Solo extraemos las claves que definimos en el fit
            # Si una clave falta, ponemos 0.0 por seguridad
            row = [float(f.get(name, 0)) for name in self.feature_names_order_]
            matrix.append(row)
        return np.array(matrix)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'means': self.means_,
                'stds': self.stds_,
                'names': self.feature_names_order_,
                'fitted': self.is_fitted
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.means_ = data['means']
            self.stds_ = data['stds']
            self.feature_names_order_ = data['names']
            self.is_fitted = data['fitted']