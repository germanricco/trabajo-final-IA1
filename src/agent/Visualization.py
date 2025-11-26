import matplotlib.pyplot as plt
import numpy as np
from typing import List

class ClusterVisualizer:
    """
    Encargada de generar representaciones gráficas de los clusters y centroides.
    """
    
    def __init__(self):
        # Colores consistentes para los clusters (0, 1, 2, 3)
        self.colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFFF33', '#33FFFF']
        self.markers = ['o', 's', '^', 'D', 'v', '<']

    def plot_clusters(self, 
                     X: np.ndarray, 
                     labels: np.ndarray, 
                     centroids: np.ndarray, 
                     feature_names: List[str],
                     feature_x: str, 
                     feature_y: str,
                     title: str = "Visualización de Clusters"):
        """
        Genera un gráfico 2D comparando dos características específicas.
        
        Args:
            X: Matriz de datos normalizados (N, features).
            labels: Array con el ID del cluster para cada punto.
            centroids: Matriz de centroides (K, features).
            feature_names: Lista con los nombres de las columnas de X.
            feature_x: Nombre de la característica para el eje X.
            feature_y: Nombre de la característica para el eje Y.
        """
        # 1. Encontrar los índices de las columnas que queremos graficar
        try:
            idx_x = feature_names.index(feature_x)
            idx_y = feature_names.index(feature_y)
        except ValueError as e:
            print(f"Error: Característica no encontrada en la lista. {e}")
            return

        plt.figure(figsize=(10, 6))
        
        # 2. Dibujar los puntos de datos (Datapoints)
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Filtrar puntos que pertenecen a este cluster
            mask = (labels == label)
            points_x = X[mask, idx_x]
            points_y = X[mask, idx_y]
            
            plt.scatter(points_x, points_y, 
                        c=self.colors[label % len(self.colors)],
                        marker=self.markers[label % len(self.markers)],
                        label=f'Cluster {label}',
                        alpha=0.6,
                        edgecolor='k')

        # 3. Dibujar los centroides (Hacerlos grandes y visibles)
        # Nota: Los centroides también tienen N dimensiones, tomamos las mismas 2.
        plt.scatter(centroids[:, idx_x], centroids[:, idx_y], 
                    c='black', 
                    marker='X', 
                    s=200, # Tamaño grande
                    linewidths=3,
                    label='Centroides')

        # Decoración
        plt.xlabel(f"{feature_x} (Normalizado)")
        plt.ylabel(f"{feature_y} (Normalizado)")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Mostrar
        plt.tight_layout()
        plt.show()