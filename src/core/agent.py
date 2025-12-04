import logging
import os
from typing import List, Dict, Optional, Any

# Importacion de subsistema de vision
from src.vision.classifier import ImageClassifier

class HardwareAgent:
    """
    Agente Central (Core).
    
    Actua como orquestador principal del sistema, integrando las capacidades de:
    1. Vision Artificial (ImageClassifier)
    2. Reconocimiento de Voz
    3. Inferencia Bayesiana

    Provee una interfaz unificada para la UI o scripts externos
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializa el agente y carga los subsistemas disponibles
        """
        # Configuración de Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("HardwareAgent")

        self.models_dir = models_dir
        
        # --- Subsistemas ---
        self.vision: Optional[ImageClassifier] = None
        self._init_vision_system()

    def _init_vision_system(self):
        """
        Inicializa y carga el modelo de vision artificial
        """
        try:
            self.vision = ImageClassifier(models_dir=self.models_dir)
            if self.vision.load_model():
                self.logger.info("✅ Sistema de visión listo.")
            else:
                self.logger.warning("⚠️ Sistema de visión NO entrenado.")
        except Exception as e:
            self.logger.error(f"Error critico inicializando subsistema de vision: {e}")

        
    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Analiza una imagen y retorna los objetos detectados con su ubicacion
        
        Este metodo orquesta el pipeline interno de vision para obtener
        datos enriquecidos (Label + BBoxes) necesarios para la UI.

        Args:
            * image_path (str): Ruta de la imagen a analizar.

        Returns:
            Lista de diccionarios con formato:
            {
                'label': 'tornillo', 'bbox': (x, y, w, h), 'confidence': ...
                ...
            }
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Archivo no encontrado: {image_path}")
            return []

        if not self.vision.is_ready:
            self.logger.error("Solicitud rechazada: Modelo de visión no cargado.")
            return []

        try:
            # Delegación simple (Responsabilidad del Clasificador)
            results = self.vision.predict(image_path)
            
            self.logger.info(f"Detección finalizada: {len(results)} objetos en {os.path.basename(image_path)}")
            return results

        except Exception as e:
            self.logger.error(f"Fallo crítico en detección: {e}")
            return []
        
    def train_vision_system(self, data_path: str = "data/raw/images/all") -> float:
        """Expone la capacidad de re-entrenamiento."""

        if not self.vision:
            self.logger.error("Intento de entrenamiento sin sistema de vision.")
            return 0.0
        
        return self.vision.train(data_path=data_path, attempts=10)


    def load_trained_model(self) -> bool:
        """
        Intenta cargar un modelo pre-entrenado desde el disco.
        Retorna True si tuvo éxito.
        """
        if self.vision.load_model():
            self.logger.info("Modelo cargado manualmente desde disco.")
            return True
        else:
            self.logger.error("No se encontró el archivo del modelo.")
            return False
        
    # =========================================================================
    # FUTURAS EXTENSIONES (Placeholders)
    # =========================================================================

    def listen_command(self) -> str:
        """
        (Futuro) Captura audio, procesa con K-NN y retorna el comando texto.
        """
        if not self.voice:
            return "MODULE_NOT_LOADED"
        # return self.voice.listen()
        pass

    def estimate_box_distribution(self, detected_objects: List[str]) -> Dict[str, float]:
        """
        (Futuro) Calcula la probabilidad Bayesiana de la caja de origen
        basada en la lista de objetos detectados.
        """
        # return self.bayes.predict(detected_objects)
        pass