import logging
import os
from collections import Counter
from typing import List, Dict, Optional, Any

# Importacion de subsistema de vision
from src.vision.classifier import ImageClassifier
# Importacion de subsistema de estimacion bayesiana
from src.analysis.estimation import BoxEstimator
# Importacion de subsistema de voz
from src.voice.voice_recognizer import VoiceRecognizer

class HardwareAgent:
    """
    Agente Central (Core).

    Actua como orquestador principal del sistema y gestor de estado, integrando:
    1. Vision Artificial (ImageClassifier)
    2. Reconocimiento de Voz
    3. Inferencia Bayesiana

    Ademas mantiene el estado de la sesion actual (conteo acumulado y ultima deteccion)
    """
    
    def __init__(self, models_dir: str = "models"):
        # Configuración de Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("HardwareAgent")
        self.models_dir = models_dir
        
        # Subsistemas
        self.vision: Optional[ImageClassifier] = None
        self.voice: Optional[VoiceRecognizer] = None
        self.box_estimator = BoxEstimator()

        # Memoria de sesión
        self._last_detections: List[Dict] = []
        self._session_total_counts: Counter = Counter()

        # Inicialización de subsistemas
        self._init_vision_system()
        self._init_voice_system()
        
    # =========================================================================
    # INICIALIZACION DE SUBSISTEMAS
    # =========================================================================

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
    

    def _init_voice_system(self):
        """
        Inicializa y carga el subsistema de voz.
        """
        try:
            # Definimos la ruta donde vivirá el modelo .pkl de voz
            voice_model_path = os.path.join(self.models_dir, "voice_model.pkl")
            
            # Instanciamos el reconocedor
            self.voice = VoiceRecognizer(model_path=voice_model_path)
            
            # Intentamos cargar el modelo si existe
            if self.voice.load_model():
                self.logger.info("✅ Sistema de voz listo y cargado.")
            else:
                self.logger.warning("⚠️ Sistema de voz inicializado pero NO entrenado.")
        except Exception as e:
            self.logger.error(f"Error crítico inicializando voz: {e}")


    # =========================================================================
    # SISTEMA DE VISION ARTIFICIAL
    # =========================================================================

    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Analiza una imagen y retorna los objetos detectados con su ubicacion
        
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
        """Re-entrena el modelo de vision"""
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
    # SISTEMA DE AUDIO
    # =========================================================================

    def listen_command(self) -> str:
        """
        Activa el micrófono, escucha y retorna el comando predicho.
        Retornos esperados: 'contar', 'salir', 'proporcion' o errores.
        Utilizado por la UI cuando se presiona el botón 'Escuchar'.
        """
        if not self.voice:
            self.logger.error("Subsistema de voz no existe.")
            return "ERROR_SYSTEM"
        
        # Retorna: 'contar', 'salir', 'proporcion' o códigos de error
        return self.voice.listen_and_predict()


    def train_voice_system(self, data_path: str = "data/raw/audio") -> bool:
        """
        Entrena el modelo KNN con los audios de la carpeta especificada.
        """
        if not self.voice:
            self.logger.error("No se puede entrenar: Subsistema de voz no inicializado.")
            return False
            
        self.logger.info(f"Solicitando entrenamiento de voz en: {data_path}")
        return self.voice.train(dataset_path=data_path)
    

    def get_count_report(self) -> str:
        """
        Genera el reporte para el comando "CONTAR".
        Devuelve texto listo para ser leído o mostrado.
        """
        if not self._last_detections:
            return "No hay ninguna muestra analizada recientemente."

        # Analizamos lo que hay en la ÚLTIMA imagen (Contexto inmediato)
        labels = [d['label'] for d in self._last_detections]
        counts = Counter(labels)
        
        # Construimos el mensaje
        msg = f"En esta muestra detecté {len(self._last_detections)} piezas.\n"
        for label, count in counts.items():
            msg += f"- {count} {label}\n"
            
        return msg

    def get_proportion_report(self) -> str:
        """
        Genera el reporte para el comando "PROPORCIÓN".
        Utiliza el estimador Bayesiano.
        """
        # Obtenemos el diccionario estructurado: 
        # {'top_box': 'A', 'confidence': 0.95, 'all_probs': {...}, ...}
        result = self.box_estimator.get_prediction()
        
        # Validación básica
        if not result:
            return "Aún no tengo datos suficientes para estimar la proporción."
            
        # Extraemos directamente los valores calculados por BoxEstimator
        top_box = result.get("top_box")
        confidence = result.get("confidence", 0.0)
        
        # Validamos que top_box no sea None (puede pasar si no hay priors)
        if top_box is None:
             return "No puedo determinar la caja con certeza todavía."

        return f"Es probable que sea la {top_box} con un {confidence*100:.1f}% de seguridad."

    
    # =========================================================================
    # LOGICA DE NEGOCIO
    # =========================================================================

    def process_sample_for_estimation(self, image_path: str) -> Dict[str, Any]:
        """
        Método principal llamado por el botón 'Analizar Muestra' de la UI.
        1. Detecta objetos en la imagen.
        2. Actualiza la memoria de la sesión.
        3. Actualiza el estimador bayesiano con los nuevos conteos.
        4. Prepara la respuesta para la UI.
        """
        # 1. Detectar objetos en la imagen
        detections = self.detect_objects(image_path)

        # 2. Actualizacion de Memoria
        self._last_detections = detections

        labels = [d['label'] for d in detections]
        current_counts = dict(Counter(labels))

        # Actualizamos acumulado total
        self._session_total_counts.update(current_counts)
        
        # 3. Actualizacion Bayesiana
        if detections:
            self.box_estimator.update(current_counts)

        # 4. Preparar respuesta para UI
        response = {
            "detections": detections,
            "count_in_image": len(detections),
            "estimation_result": self.box_estimator.get_prediction()
        }
        
        return response

    def reset_estimation(self):
        """
        Llamado por el botón 'Reiniciar Lote.
        """
        self.box_estimator.reset()
        self._last_detections = []
        self._session_total_counts = Counter()
        self.logger.info("Estimación y Contadores de sesion reiniciados.")