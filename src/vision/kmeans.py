import numpy as np
import logging

class KMeansModel:
    """
    Implementación de K-Means 
    Diseñado para trabajar con matrices NumPy normalizadas (Z-Score).
    """
    
    def __init__(self,
                 n_clusters: int = 4,
                 max_iters: int = 100,
                 tol: float = 0.0001,
                 n_init: int = 10):
        """
        Configuración del hiper-parámetros del modelo.
        
        Args:
            * n_clusters (K): Número de grupos a formar (ej: 4 para tornillos, clavos, etc).
            * max_iters: Límite de seguridad para evitar bucles infinitos.
            * tol: Tolerancia de convergencia. Si los centroides se mueven menos que esto, paramos.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        
        # Estado del modelo: Aquí guardaremos las coordenadas de los K centroides
        # Forma esperada: (n_clusters, n_features) -> Ej: (4, 9)
        self.centroids = None
        self.inertia_ = None  # Suma de distancias al cuadrado de puntos a sus centroides
        
        self.logger = logging.getLogger(__name__)

    def fit(self, X: np.ndarray):
        """
        Entrena el modelo buscando los centroides óptimos.
        
        Args:
            X: Matriz de datos normalizados. Forma: (n_samples, n_features).
               Ej: (16 objetos, 9 características)
        """
        n_samples, n_features = X.shape
        
        # Validación: No podemos buscar 4 grupos si solo hay 2 datos
        if n_samples < self.n_clusters:
            raise ValueError(f"Datos insuficientes ({n_samples}) para {self.n_clusters} clusters.")

        self.logger.info(f"🧠 Iniciando K-Means con K={self.n_clusters} sobre {n_samples} muestras.")

        # Variables para guardar el mejor modelo encontrado en los n_init intentos
        best_inertia = np.inf
        best_centroids = None

        for run in range(self.n_init):
            # PASO 1: Inicialización
            current_centroids = self._initialize_centroids(X)

            for i in range(self.max_iters):
                # Guardamos copia para comparar luego cuánto se movieron
                old_centroids = current_centroids.copy()

                # Paso 2 - Asignacion usando centroides actuales
                distances = self._calculate_distances_with_centroids(X, current_centroids)

                # Para cada punto, encontramos el índice del centroide más cercano (0, 1, 2 o 3)
                labels = np.argmin(distances, axis=1)

                # PASO 3: Actualización (Maximization Step)
                # Movemos los centroides al promedio de sus puntos asignados
                current_centroids = self._update_centroids_logic(X, labels, current_centroids)

                # PASO 4: Chequeo de Convergencia
                # Calculamos la distancia total que se movieron los centroides
                shift = np.linalg.norm(current_centroids - old_centroids)

                # Log de progreso para ver cómo baja el movimiento
                if i % 5 == 0: 
                    self.logger.debug(f"Iteración {i}: Movimiento = {shift:.6f}")

                if shift < self.tol:
                    self.logger.debug(f"✅ Convergencia alcanzada en iteración {i}. Movimiento: {shift:.6f}")
                    break
            
            # --- Fin de la ejecución. Calculamos la nota (Inercia) ---
            current_inertia = self._calculate_inertia(X, current_centroids)

            # Si esta ejecución es mejor que la mejor que teníamos, la guardamos
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centroids = current_centroids
                self.logger.debug(f"Run {run}: Nueva mejor inercia: {best_inertia:.2f}")

        # Al final, nos quedamos con el campeón del torneo
        self.centroids = best_centroids
        self.inertia_ = best_inertia
        self.logger.info(f"✅ Entrenamiento finalizado. Mejor inercia: {self.inertia_:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Inferencia: Asigna nuevos datos a los clusters ya aprendidos.
        No modifica los centroides.
        """
        if self.centroids is None:
            raise RuntimeError("El modelo no está entrenado. Ejecuta fit() primero.")
            
        # Simplemente calculamos distancias y elegimos el menor
        # Retornamos los índices de los clusters asignados
        distances = self._calculate_distances_with_centroids(X, self.centroids)
        return np.argmin(distances, axis=1)


    # =========================================================================
    # MÉTODOS MATEMÁTICOS (TU TAREA: IMPLEMENTAR LA LÓGICA AQUÍ)
    # =========================================================================

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Selecciona K puntos aleatorios del dataset X para ser los centroides iniciales.
        """
        # Tomo los indices aleatorios sin reemplazo para no duplicar centroides
        #! Metodo Forgy al elegir puntos reales como centroides iniciales
        idx = np.random.choice(a=X.shape[0], size=self.n_clusters, replace=False)

        initial_centroids = X[idx]
        return initial_centroids
    
    def _calculate_inertia(self, X: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calcula la Suma de Errores Cuadráticos (SSE).
        Inercia = Suma(distancia_al_centroide_mas_cercano ^ 2)
        """
        # 1. Calculamos distancias a todos los centroides
        distances = self._calculate_distances_with_centroids(X, centroids)
        
        # 2. Nos quedamos solo con la distancia al centroide elegido (el mínimo de cada fila)
        min_distances = np.min(distances, axis=1)
        
        # 3. Elevamos al cuadrado y sumamos todo
        inertia = np.sum(min_distances ** 2)
        return inertia

    def _calculate_distances_with_centroids(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Refactorizamos para poder pasar 'centroids' como argumento,
        así sirve tanto para 'self.centroids' como para 'current_centroids' temporalmente.

        Calcula la distancia Euclidiana de cada punto a cada centroide.
        
        Args:
            * X: (N, Features)
            * Centroids: (K, Features)
            
        Returns:
            Matriz de distancias de forma (N, K)
        """

        # Calculo la distancia entre cada data-point X y cada centroide (x-c)
        # Cada elemento de matriz ahora es (x - c) para cada combinación de punto y centroide
        #! Broadcasting
        # X shape: (N, Features)
        # X[:,np.newaxis] -> (N, 1, Features)
        # self.centroids -> (K, Features)
        # diff_matrix -> (N, K, Features)
        diff_matrix = X[:, np.newaxis] - centroids

        distances = np.sqrt(np.sum(diff_matrix ** 2, axis=2))

        return distances

    def _update_centroids_logic(self,
                                X: np.ndarray,
                                labels: np.ndarray,
                                current_centroids: np.ndarray) -> np.ndarray:
        """
        Recalcula la posición de los centroides como el promedio de los puntos.
        """
        new_centroids = np.zeros_like(current_centroids)
        
        for k in range(self.n_clusters):
            # Puntos asignados al cluster k
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                # Calculamos media de los puntos asignados
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Manejo de Clusters Vacios. Reinicializamos el centroide aleatoriamente
                random_idx = np.random.choice(X.shape[0])
                new_centroids[k] = X[random_idx]

        return new_centroids