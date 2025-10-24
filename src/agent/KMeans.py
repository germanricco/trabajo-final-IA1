import numpy as np
import os
import cv2
from typing import Union, List, Tuple, Dict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.agent.FeatureExtractor import FeatureExtractor
import pickle


class KMeans:
    def __init__(self,
                 n_clusters: int = 4,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 random_state: Union[int, None] = None,
                 normalizar: bool = True  # Normalización crítica
                 ):
        """
        Constructor del algoritmo K-Means para clasificacion de piezas
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.normalizar = normalizar

        # Atributos para normalización
        self.media_ = None
        self.desviacion_ = None

        # Atributos que se estableceran durante el ajuste
        self.centroides = None
        self.labels = None
        self.inercia_ = None
        self.n_iter_ = 0
        self.historial_centroides = []

        if random_state is not None:
            np.random.seed(random_state)
        
    def _normalizar_datos(self, X: np.ndarray) -> np.ndarray:
        """Normaliza los datos para que todas las características contribuyan equitativamente"""
        if self.media_ is None:
            self.media_ = np.mean(X, axis=0)
            self.desviacion_ = np.std(X, axis=0)
            # Evitar división por cero
            self.desviacion_[self.desviacion_ == 0] = 1.0
        
        return (X - self.media_) / self.desviacion_
    
    def _desnormalizar_centroides(self) -> np.ndarray:
        """Desnormaliza los centroides para interpretación"""
        if self.media_ is None or self.desviacion_ is None:
            return self.centroides
        return self.centroides * self.desviacion_ + self.media_
    
    def _inicializar_centroides(self, puntos: np.ndarray) -> np.ndarray:
        """
        Inicialización mejorada: k-means++ para mejor convergencia
        Referencia: https://en.wikipedia.org/wiki/K-means%2B%2B
        """
        n_puntos, n_caracteristicas = puntos.shape
        
        # Paso 1: Elegir primer centroide aleatoriamente
        centroides = [puntos[np.random.randint(n_puntos)]]
        
        # Paso 2: Elegir centros restantes con probabilidad proporcional a distancia^2
        for _ in range(1, self.n_clusters):
            distancias = np.array([min([np.linalg.norm(p - c)**2 for c in centroides]) for p in puntos])
            probabilidades = distancias / distancias.sum()
            indices_acumulados = np.cumsum(probabilidades)
            r = np.random.rand()
            
            for i, prob_acum in enumerate(indices_acumulados):
                if r < prob_acum:
                    centroides.append(puntos[i])
                    break
        
        return np.array(centroides)
    
    def _calcular_distancias(self, puntos: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        """Calcula distancias euclidianas - tu implementación es óptima"""
        return np.sqrt(np.sum((puntos[:, np.newaxis, :] - centroides[np.newaxis, :, :]) ** 2, axis=2))

    def _asignar_clusters(self, puntos: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        return np.argmin(self._calcular_distancias(puntos, centroides), axis=1)

    def _actualizar_centroides(self, puntos: np.ndarray, labels: np.ndarray) -> np.ndarray:
        nuevos_centroides = np.zeros((self.n_clusters, puntos.shape[1]))
        for i in range(self.n_clusters):
            puntos_cluster = puntos[labels == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides[i] = np.mean(puntos_cluster, axis=0)
            else:
                # Mejor estrategia: reinicializar lejos de centros existentes
                print(f"⚠️  Cluster {i} vacío, reinicializando...")
                nuevos_centroides[i] = puntos[np.random.randint(0, len(puntos))]
        return nuevos_centroides

    def _calcular_inercia(self, puntos: np.ndarray, labels: np.ndarray, centroides: np.ndarray) -> float:
        inercia = 0.0
        for i in range(self.n_clusters):
            puntos_cluster = puntos[labels == i]
            if len(puntos_cluster) > 0:
                distancias = np.linalg.norm(puntos_cluster - centroides[i], axis=1)
                inercia += np.sum(distancias ** 2)
        return inercia

    def ajustar(self, puntos: np.ndarray) -> 'KMeans':
        """Ajusta el modelo con normalización y mejor inicialización"""
        # Validación
        if puntos.shape[0] < self.n_clusters:
            raise ValueError(f"Insuficientes puntos ({puntos.shape[0]}) para {self.n_clusters} clusters")
        
        # Normalización CRÍTICA para nuestras características
        if self.normalizar:
            puntos = self._normalizar_datos(puntos)
        
        self.puntos = puntos.copy()
        
        # Inicialización mejorada
        self.centroides = self._inicializar_centroides(puntos)
        self.historial_centroides = [self.centroides.copy()]
        
        print(f"🚀 Iniciando K-Means para {self.n_clusters} tipos de piezas")
        print(f"📊 Dataset: {puntos.shape[0]} muestras, {puntos.shape[1]} características")
        
        for iteracion in range(self.max_iter):
            self.labels = self._asignar_clusters(puntos, self.centroides)
            nuevos_centroides = self._actualizar_centroides(puntos, self.labels)
            
            movimiento = np.linalg.norm(nuevos_centroides - self.centroides)
            self.centroides = nuevos_centroides
            self.historial_centroides.append(self.centroides.copy())
            
            inercia_actual = self._calcular_inercia(puntos, self.labels, self.centroides)
            
            if iteracion % 10 == 0:
                print(f"🔄 Iteración {iteracion + 1}: Inercia = {inercia_actual:.4f}, Movimiento = {movimiento:.6f}")
            
            if movimiento < self.tol:
                print(f"✅ Convergencia en {iteracion + 1} iteraciones")
                break
        
        self.n_iter_ = iteracion + 1
        self.inercia_ = self._calcular_inercia(puntos, self.labels, self.centroides)
        
        print(f"🎯 Ajuste completado")
        print(f"📈 Inercia final: {self.inercia_:.4f}")
        print(f"🔄 Iteraciones totales: {self.n_iter_}")
        
        return self

    def predecir(self, puntos: np.ndarray) -> np.ndarray:
        """Predice clusters para nuevos datos (con normalización consistente)"""
        if self.centroides is None:
            raise ValueError("Modelo no ajustado")
        
        if self.normalizar:
            puntos = (puntos - self.media_) / self.desviacion_
        
        return self._asignar_clusters(puntos, self.centroides)

    # NUEVOS MÉTODOS para evaluación y visualización
    def evaluar_clustering(self, etiquetas_reales: np.ndarray) -> Dict:
        """
        Evalúa el clustering comparando con etiquetas reales
        Referencia: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(etiquetas_reales, self.labels)
        nmi = normalized_mutual_info_score(etiquetas_reales, self.labels)
        
        return {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'inercia': self.inercia_,
            'n_iteraciones': self.n_iter_
        }

    def visualizar_clusters_2d(self, caracteristicas: List[str], indice_caracteristica_x: int = 0, 
                             indice_caracteristica_y: int = 1, etiquetas_reales: np.ndarray = None):
        """
        Visualización 2D de los clusters (proyección)
        """
        if self.puntos.shape[1] < 2:
            print("❌ Se necesitan al menos 2 características para visualización 2D")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Clusters predichos
        plt.subplot(1, 2, 1)
        for i in range(self.n_clusters):
            puntos_cluster = self.puntos[self.labels == i]
            plt.scatter(puntos_cluster[:, indice_caracteristica_x], 
                       puntos_cluster[:, indice_caracteristica_y], 
                       label=f'Cluster {i}', alpha=0.7)
        
        centroides_desnorm = self._desnormalizar_centroides()
        plt.scatter(centroides_desnorm[:, indice_caracteristica_x], 
                   centroides_desnorm[:, indice_caracteristica_y], 
                   marker='X', s=200, c='black', label='Centroides')
        
        plt.xlabel(f'{caracteristicas[indice_caracteristica_x]}')
        plt.ylabel(f'{caracteristicas[indice_caracteristica_y]}')
        plt.title('Clusters K-Means')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Etiquetas reales (si disponibles)
        if etiquetas_reales is not None:
            plt.subplot(1, 2, 2)
            clases_reales = np.unique(etiquetas_reales)
            for clase in clases_reales:
                puntos_clase = self.puntos[etiquetas_reales == clase]
                plt.scatter(puntos_clase[:, indice_caracteristica_x], 
                           puntos_clase[:, indice_caracteristica_y], 
                           label=f'Clase Real {clase}', alpha=0.7)
            
            plt.xlabel(f'{caracteristicas[indice_caracteristica_x]}')
            plt.ylabel(f'{caracteristicas[indice_caracteristica_y]}')
            plt.title('Etiquetas Reales')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def guardar_modelo(self, ruta_archivo: str):
        """
        Guarda el modelo K-Means entrenado en un archivo.
        
        Args:
            ruta_archivo: Ruta donde guardar el modelo (.pkl)
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        
        # Datos a guardar (solo lo necesario para predecir)
        modelo_data = {
            'centroides': self.centroides,
            'media_': self.media_,
            'desviacion_': self.desviacion_,
            'n_clusters': self.n_clusters,
            'normalizar': self.normalizar,
            'feature_names': getattr(self, 'feature_names', None)
        }
        
        with open(ruta_archivo, 'wb') as f:
            pickle.dump(modelo_data, f)
        
        print(f"💾 Modelo guardado en: {ruta_archivo}")

    @classmethod
    def cargar_modelo(cls, ruta_archivo: str) -> 'KMeans':
        """
        Carga un modelo K-Means desde archivo.
        
        Args:
            ruta_archivo: Ruta del modelo guardado
            
        Returns:
            Instancia de KMeans lista para predecir
        """
        with open(ruta_archivo, 'rb') as f:
            modelo_data = pickle.load(f)
        
        # Crear instancia
        kmeans = cls(
            n_clusters=modelo_data['n_clusters'],
            normalizar=modelo_data['normalizar']
        )
        
        # Restaurar estado entrenado
        kmeans.centroides = modelo_data['centroides']
        kmeans.media_ = modelo_data['media_']
        kmeans.desviacion_ = modelo_data['desviacion_']
        
        if 'feature_names' in modelo_data:
            kmeans.feature_names = modelo_data['feature_names']
        
        print(f"📂 Modelo cargado desde: {ruta_archivo}")
        return kmeans
    
    def predecir_imagen_individual(self, extractor: FeatureExtractor, 
                                 bounding_boxes: List[Tuple], 
                                 masks: List[np.ndarray]) -> List[Dict]:
        """
        Realiza predicción completa para una imagen individual.
        
        Args:
            extractor: Instancia de FeatureExtractor
            bounding_boxes: Bounding boxes de la segmentación
            masks: Máscaras de la segmentación
            
        Returns:
            Lista de diccionarios con predicciones para cada objeto
        """
        # Extraer características
        features = extractor.extract_features(bounding_boxes, masks)
        
        if not features:
            return []
        
        # Convertir a matriz de características
        X = np.array([extractor.get_feature_vector(f) for f in features])
        
        # Predecir clusters
        clusters = self.predecir(X)
        
        # Combinar características con predicciones
        resultados = []
        for i, (feature_dict, cluster) in enumerate(zip(features, clusters)):
            resultado = {
                'object_id': i + 1,
                'cluster': int(cluster),
                'clase_asignada': self._mapear_cluster_a_clase(cluster),
                'bounding_box': feature_dict['bounding_box'],
                'caracteristicas': {
                    'circularity': feature_dict['circularity'],
                    'aspect_ratio': feature_dict['aspect_ratio'],
                    'solidity': feature_dict['solidity'],
                    'perimeter_area_ratio': feature_dict['perimeter_area_ratio']
                },
                'confianza': self._calcular_confianza(feature_dict, cluster)
            }
            resultados.append(resultado)
        
        return resultados
    
    def _mapear_cluster_a_clase(self, cluster: int) -> str:
        """
        Mapea número de cluster a nombre de clase.
        NOTA: Esto requiere calibración con datos etiquetados
        """
        mapeo = {
            0: "Tuerca",
            1: "Arandela", 
            2: "Tornillo",
            3: "Clavo"
        }
        return mapeo.get(cluster, "Desconocido")

    def _calcular_confianza(self, features: Dict, cluster: int) -> float:
        """
        Calcula una medida de confianza basada en distancia al centroide.
        """
        vector = np.array([
            features['circularity'],
            features['aspect_ratio'],
            features['solidity'],
            features['perimeter_area_ratio'],
            features['hu_moment_1'],
            features['hu_moment_2']
        ])
        
        if self.normalizar:
            vector = (vector - self.media_) / self.desviacion_
        
        distancia = np.linalg.norm(vector - self.centroides[cluster])
        # Convertir distancia a confianza (inversa)
        confianza = 1.0 / (1.0 + distancia)
        return float(confianza)