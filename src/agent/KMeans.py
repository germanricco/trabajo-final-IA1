import numpy as np
import os
import cv2
from typing import Union, List, Tuple, Dict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.agent.FeatureExtractor import FeatureExtractor
import pickle
import logging


class KMeans:
    def __init__(self,
                 n_clusters: int = 4,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 random_state: Union[int, None] = None,
                 normalizar: bool = True
                 ):
        """
        Algoritmo K-Means para clasificacion de piezas

        Args:
            * n_clusters: Número de clusters (tipos de piezas)
            * max_iter: Número máximo de iteraciones
            * tol: Tolerancia para convergencia
            * random_state: Semilla para reproducibilidad
            * normalizar: Si normalizar características antes de ajustar
        """
        # Atributos de inicializacion
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
        self.historial_inercia = []

        if random_state is not None:
            np.random.seed(random_state)

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"KMeans inicializado con {n_clusters} clusters, max_iter={max_iter}, tol={tol}, normalizar={normalizar}")

    def _normalizar_datos(self, X: np.ndarray) -> np.ndarray:
        """
        Normaliza los datos usando Z-score normalization

        Args:
            * X: Matriz de datos [n_samples, n_features]
        
        Returns:
            * X_normalized: Datos normalizados
        """
        if self.media_ is None or self.desviacion_ is None:
            self.media_ = np.mean(X, axis=0)
            self.desviacion_ = np.std(X, axis=0)
            # Evitar división por cero
            self.desviacion_[self.desviacion_ == 0] = 1.0
        
        X_normalizado = (X - self.media_) / self.desviacion_
        return X_normalizado
    
    def _inicializar_centroides(self, X: np.ndarray) -> np.ndarray:
        """
        Inicializa los centroides usando K-Means++

        Args:
            * X: Matriz de datos [n_samples, n_features]
        
        Returns:
            * centroides: Matriz de centroides [n_clusters, n_features]
        """
        n_muestras, n_caracteristicas = X.shape
        centroides = np.zeros((self.n_clusters, n_caracteristicas))

        # Primer Centroide aleatorio
        primer_indice = np.random.randint(n_muestras)
        centroides[0] = X[primer_indice]
        
        # Resto de centroides usando K-Means++
        for i in range(1, self.n_clusters):
            # Calcular distancias al centroide más cercano
            distancias = np.array([min([np.linalg.norm(x - c)**2 for c in centroides[:i]]) 
                                 for x in X])
            
            # Probabilidad proporcional a la distancia al cuadrado
            probabilidades = distancias / np.sum(distancias)

            # Elegir siguiente centroide basado en probabilidades
            indice_elegido = np.random.choice(n_muestras, p=probabilidades)
            centroides[i] = X[indice_elegido]
            
        self.logger.debug(f"Centroides inicializados con K-Means++")
        return centroides
    
    def _calcular_distancias(self, X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia entre cada punto y cada centroide.
        
        Args:
            * X: Matriz de características [n_muestras, n_caracteristicas]
            * centroides: Matriz de centroides [n_clusters, n_caracteristicas]
            
        Returns:
            * distancias: Matriz de distancias [n_muestras, n_clusters]
        """
        n_muestras = X.shape[0]
        n_clusters = centroides.shape[0]
        distancias = np.zeros((n_muestras, n_clusters))
        
        for i in range(n_clusters):
            # Distancia euclidiana al cuadrado (más eficiente sin sqrt)
            distancias[:, i] = np.sum((X - centroides[i])**2, axis=1)
        
        return distancias
    
    def _asignar_clusters(self, X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        """
        Asigna cada punto al cluster del centroide más cercano.
        
        Args:
            * X: Array de características [n_muestras, n_caracteristicas]
            * centroides: Array de centroides [n_clusters, n_caracteristicas]
            
        Returns:
            * labels: Array de etiquetas de cluster para cada punto [n_muestras]
        """
        distancias = self._calcular_distancias(X, centroides)
        labels = np.argmin(distancias, axis=1)
        return labels

    def _actualizar_centroides(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Actualiza los centroides como la media de los puntos en cada cluster.
        
        Args:
            * X: Array de características [n_muestras, n_caracteristicas]
            * labels: Array de etiquetas de cluster [n_muestras]
            
        Returns:
            * nuevos_centroides: Array de centroides actualizados [n_clusters, n_caracteristicas]
        """
        n_clusters = self.n_clusters
        n_caracteristicas = X.shape[1]
        nuevos_centroides = np.zeros((n_clusters, n_caracteristicas))
        
        for i in range(n_clusters):
            puntos_cluster = X[labels == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides[i] = np.mean(puntos_cluster, axis=0)
            else:
                # Si un cluster está vacío, reinicializar aleatoriamente
                nuevos_centroides[i] = X[np.random.randint(X.shape[0])]
        
        return nuevos_centroides

    def _calcular_inercia(self, puntos: np.ndarray, labels: np.ndarray, centroides: np.ndarray) -> float:
        """
        Calcula la inercia total (suma de distancias al cuadrado dentro de clusters).
        
        Args:
            * X: Array de características [n_muestras, n_caracteristicas]
            * labels: Array de etiquetas de cluster [n_muestras]
            * centroides: Array de centroides [n_clusters, n_caracteristicas]
            
        Returns:
            * inercia: Suma total de distancias al cuadrado
        """
        inercia = 0.0
        for i in range(self.n_clusters):
            puntos_cluster = puntos[labels == i]
            if len(puntos_cluster) > 0:
                distancias = np.sum((puntos_cluster - centroides[i])**2)
                inercia += distancias
        return inercia

    def ajustar(self, X: np.ndarray) -> 'KMeans':
        """
        Entrena el modelo K-Means con los datos proporcionados.
        
        Args:
            * X: Matriz de características [n_muestras, n_caracteristicas]
            
        Returns:
            * self: Instancia entrenada
        """
        self.logger.info("Iniciando entrenamiento de K-Means")
        
        # Validar datos de entrada
        X = np.array(X, dtype=float)
        if len(X.shape) != 2:
            raise ValueError("X debe ser una matriz 2D [n_muestras, n_caracteristicas]")
        
        # Normalizar datos si está configurado
        if self.normalizar:
            X = self._normalizar_datos(X)
        
        n_muestras, n_caracteristicas = X.shape
        
        # Inicializar centroides
        self.centroides = self._inicializar_centroides(X)
        self.historial_centroides.append(self.centroides.copy())
        
        # Bucle principal de entrenamiento
        for iteracion in range(self.max_iter):
            self.n_iter_ = iteracion + 1
            
            # Asignar clusters
            self.labels = self._asignar_clusters(X, self.centroides)
            
            # Calcular inercia actual
            inercia_actual = self._calcular_inercia(X, self.labels, self.centroides)
            self.historial_inercia.append(inercia_actual)
            
            # Actualizar centroides
            nuevos_centroides = self._actualizar_centroides(X, self.labels)
            
            # Calcular desplazamiento de centroides
            desplazamiento = np.max(np.linalg.norm(nuevos_centroides - self.centroides, axis=1))
            
            # Actualizar centroides
            self.centroides = nuevos_centroides
            self.historial_centroides.append(self.centroides.copy())
            
            self.logger.debug(f"Iteración {iteracion + 1}: Inercia = {inercia_actual:.4f}, "
                            f"Desplazamiento máximo = {desplazamiento:.6f}")
            
            # Verificar convergencia
            if desplazamiento < self.tol:
                self.logger.info(f"Convergencia alcanzada en iteración {iteracion + 1}")
                break
        
        # Calcular inercia final
        self.inercia_ = self._calcular_inercia(X, self.labels, self.centroides)
        
        self.logger.info(f"Entrenamiento completado. Inercia final: {self.inercia_:.4f}")
        return self
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Predice los clusters para nuevos datos.
        
        Args:
            * X: Matriz de características [n_muestras, n_caracteristicas]
            
        Returns:
            * labels: Array de etiquetas predichas [n_muestras]
        """
        if self.centroides is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        # Normalizar datos si el modelo fue entrenado con normalización
        if self.normalizar:
            X = (X - self.media_) / self.desviacion_
        
        labels = self._asignar_clusters(X, self.centroides)
        return labels
    

    def guardar_modelo(self, ruta: str) -> None:
        """
        Guarda el modelo entrenado en un archivo.
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        with open(ruta, 'wb') as archivo:
            pickle.dump(self, archivo)
        self.logger.info(f"Modelo guardado en: {ruta}")

    @classmethod
    def cargar_modelo(cls, ruta: str) -> 'KMeans':
        """
        Carga un modelo entrenado desde un archivo.
        
        Args:
            ruta: Ruta del archivo del modelo
            
        Returns:
            modelo: Instancia de KMeans cargada
        """
        with open(ruta, 'rb') as archivo:
            modelo = pickle.load(archivo)
        logging.getLogger(__name__).info(f"Modelo cargado desde: {ruta}")
        return modelo

    def visualizar_clusters_3d(self, X: np.ndarray, 
                             caracteristicas: List[str] = None,
                             titulo: str = "Clusters K-Means",
                             guardar: bool = False,
                             ruta_guardado: str = None) -> None:
        """
        Visualiza los clusters en 3D usando las primeras 3 características.
        
        Args:
            * X: Datos originales [n_muestras, n_caracteristicas]
            * caracteristicas: Nombres de las características para los ejes
            * titulo: Título del gráfico
            * guardar: Si guardar la visualización
            * ruta_guardado: Ruta donde guardar la imagen
        """
        if X.shape[1] < 3:
            self.logger.error("No se pueden visualizar en 3D con menos de 3 características")
            raise ValueError("Se necesitan al menos 3 características para visualización 3D")
        
        # Predecir clusters si no están asignados
        if self.labels is None:
            labels = self.predecir(X)
        else:
            labels = self.labels
        
        # Crear figura 3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Colores para clusters
        colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Graficar puntos
        for i in range(self.n_clusters):
            puntos_cluster = X[labels == i]
            if len(puntos_cluster) > 0:
                ax.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], puntos_cluster[:, 2],
                          c=colores[i % len(colores)], label=f'Cluster {i}', alpha=0.7, s=50)
        
        # Graficar centroides
        centroides_originales = self.centroides
        if self.normalizar:
            # Desnormalizar centroides para visualización
            centroides_originales = self.centroides * self.desviacion_ + self.media_
        
        ax.scatter(centroides_originales[:, 0], centroides_originales[:, 1], 
                  centroides_originales[:, 2], marker='X', s=200, c='black', 
                  label='Centroides', linewidths=2)
        
        # Configurar ejes
        if caracteristicas is not None:
            ax.set_xlabel(caracteristicas[0], fontsize=12)
            ax.set_ylabel(caracteristicas[1], fontsize=12)
            ax.set_zlabel(caracteristicas[2], fontsize=12)
        else:
            ax.set_xlabel('Característica 1', fontsize=12)
            ax.set_ylabel('Característica 2', fontsize=12)
            ax.set_zlabel('Característica 3', fontsize=12)
        
        ax.set_title(titulo, fontsize=14)
        ax.legend()
        
        # Guardar si es necesario
        if guardar:
            if ruta_guardado is None:
                ruta_guardado = f"kmeans_clusters_3d_{self.n_clusters}.png"
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualización guardada en: {ruta_guardado}")
        
        plt.show()

    def visualizar_evolucion(self, guardar: bool = False, 
                           ruta_guardado: str = None) -> None:
        """
        Visualiza la evolución de la inercia durante el entrenamiento.
        
        Args:
            guardar: Si guardar la visualización
            ruta_guardado: Ruta donde guardar la imagen
        """
        if not self.historial_inercia:
            raise ValueError("No hay historial de entrenamiento disponible")
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.historial_inercia) + 1), self.historial_inercia, 
                marker='o', linewidth=2, markersize=6)
        plt.xlabel('Iteración', fontsize=12)
        plt.ylabel('Inercia', fontsize=12)
        plt.title('Evolución de la Inercia durante el Entrenamiento', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if guardar:
            if ruta_guardado is None:
                ruta_guardado = f"kmeans_evolucion_inercia_{self.n_clusters}.png"
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            self.logger.info(f"Gráfico de evolución guardado en: {ruta_guardado}")
        
        plt.show()

    def obtener_metricas(self) -> dict:
        """
        Retorna métricas importantes del modelo.
        
        Returns:
            metricas: Diccionario con métricas del modelo
        """
        return {
            'n_clusters': self.n_clusters,
            'inercia': self.inercia_,
            'n_iteraciones': self.n_iter_,
            'normalizado': self.normalizar,
            'convergio_early': self.n_iter_ < self.max_iter
        }

    def __str__(self) -> str:
        """Representación en string del modelo"""
        metricas = self.obtener_metricas()
        return (f"KMeans(n_clusters={metricas['n_clusters']}, "
                f"inercia={metricas['inercia']:.4f}, "
                f"iteraciones={metricas['n_iteraciones']})")