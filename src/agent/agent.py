import logging
from src.agent.ImageClassifier import ImageClassifier

# Placeholder para el futuro
# from src.agent.VoiceRecognizer import VoiceRecognizer

class HardwareAgent:
    """
    Agente Principal (Fachada).
    Centraliza las capacidades cognitivas del sistema (Ver, Escuchar, Razonar).
    """
    
    def __init__(self):
        # Configuración de logs
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("HardwareAgent")
        
        # --- Inicialización de Sentidos ---
        
        # 1. Visión (Tu clasificador de hardware)
        self.vision = ImageClassifier()
        
        # 2. Voz (Futura implementación)
        self.voice = None 
        # self.voice = VoiceRecognizer()
        
        # Intentar cargar memoria previa
        if self.vision.load_model():
            self.logger.info("Sistema de visión cargado y listo.")
        else:
            self.logger.warning("Sistema de visión no entrenado.")

    def train(self, data_path="data/raw/images/all"):
        """
        Reentrena las capacidades del agente.
        Por ahora solo visión, en el futuro podría reentrenar comandos de voz.
        """
        self.logger.info("Iniciando secuencia de entrenamiento...")
        
        # Entrenar visión
        accuracy = self.vision.train(data_path)
        
        return {
            "vision_accuracy": accuracy,
            "status": "Entrenamiento completado"
        }

    def identify_object(self, image_path):
        """
        Utiliza el sistema de visión para identificar qué hay en la imagen.
        """
        try:
            results = self.vision.predict_single_image(image_path)
            
            if not results:
                return "No detecté ningún objeto conocido."
            
            # Formatear respuesta amigable
            # Ej: "Veo 2 tornillos y 1 arandela"
            from collections import Counter
            counts = Counter(results)
            description = ", ".join([f"{cnt} {name}" for name, cnt in counts.items()])
            
            return f"Detectado: {description}"
            
        except RuntimeError:
            return "Error: Mis ojos no están calibrados (Modelo no entrenado)."
        except Exception as e:
            self.logger.error(f"Error en identificación: {e}")
            return "Ocurrió un error al intentar ver la imagen."