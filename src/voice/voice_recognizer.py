import os
import pickle
import numpy as np
import sounddevice as sd
import logging
from typing import Optional, Dict, Union

# Importamos las clases que ya definimos
from src.voice.audio_feature_extractor import AudioFeatureExtractor
from src.voice.knn_model import KNNModel

class VoiceRecognizer:
    """
    Orquestador del sistema de voz.
    Responsabilidades:
    1. Interacción con Hardware (Micrófono).
    2. Coordinación del pipeline: Audio -> Extractor -> KNN.
    3. Gestión de entrenamiento masivo desde carpetas.
    """

    def __init__(self,
                 model_path: str = "models/voice_model.pkl",
                 sample_rate: int = 16000,
                 duration: int = 2,
                 k_neighbors: int = 5):
        self.logger = logging.getLogger("VoiceRecognizer")
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Instancias de los submodulos
        self.knn = KNNModel(k=k_neighbors)
        self.extractor = AudioFeatureExtractor(sr=sample_rate)

        # Estado
        self.is_ready = False

    def load_model(self) -> bool:
        """
        Carga el modelo.
        """
        # Delegación: El KNN sabe cómo cargarse a sí mismo
        success = self.knn.load(self.model_path)
        
        if success:
            self.is_ready = True
            self.logger.info(f"✅ Modelo de voz cargado desde: {self.model_path}")
        else:
            self.logger.warning(f"⚠️ No se pudo cargar el modelo de voz en: {self.model_path}")
            self.is_ready = False
            
        return success


    def save_model(self):
        """
        Guarda el modelo.
        """
        self.knn.save(self.model_path)
        self.logger.info(f"💾 Modelo guardado exitosamente en: {self.model_path}")


    def train(self, dataset_path: str) -> bool:
        """
        Recorre el dataset, extrae características y entrena el KNN.

        Args:
            * dataset_path (str): Ruta a la carpeta del dataset.

        Returns:
            * bool: True si el entrenamiento fue exitoso.
        """
        self.logger.info(f"🚀 Iniciando entrenamiento de voz desde: {dataset_path}")
        X = []
        y = []

        if not os.path.exists(dataset_path):
            self.logger.error("❌ Directorio de dataset no encontrado.")
            return False

        # Obtener subcarpetas (clases)
        classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        if not classes:
            self.logger.error("❌ No se encontraron carpetas de clases en el dataset.")
            return False
        
        # Recorrer carpetas
        count_files = 0
        for label in classes:
            # Obtener path a la carpeta de la clase
            folder_path = os.path.join(dataset_path, label)
            # Listar archivos .wav
            wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

            self.logger.info(f"   📂 Procesando clase '{label}' ({len(wav_files)} audios)...")
            
            # Recorrer archivos .wav
            for wav in wav_files:
                file_path = os.path.join(folder_path, wav)
                # Extraer características
                features = self.extractor.extract_features(file_path)
                # Agregar features y etiqueta si es válido
                if features is not None:
                    X.append(features)
                    y.append(label)
                    # Aumentar contador
                    count_files += 1

        if count_files == 0:
            self.logger.error("❌ No se generaron características válidas. Revise los audios.")
            return False

        # Entrenamiento y persistencia
        self.knn.fit(X, y)
        self.is_ready = True
        self.save_model()
        
        self.logger.info(f"✅ Entrenamiento finalizado. Total muestras: {count_files}")
        return True


    def listen_and_predict(self) -> str:
        """
        Captura audio del micrófono y retorna el comando predicho.

        Returns:
            * str: Etiqueta del comando predicho o código de error.
        """
        if not self.is_ready:
            self.logger.error("Solicitud de voz rechazada: Modelo no cargado.")
            return "ERROR_MODEL_NOT_LOADED"

        try:
            self.logger.info(f"🎤 Escuchando ({self.duration}s)...")
            
            # 1. Grabación (Bloqueante)
            audio_data = sd.rec(
                int(self.duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=1, 
                dtype='float32'
            )
            sd.wait() # Esperar a que termine la grabación
            
            # Aplanar array (de [[v],[v]] a [v,v])
            audio_data = audio_data.flatten()

            # 2. Extracción de características
            features = self.extractor.extract_features(audio_data)
            
            if features is None:
                self.logger.warning("Ruido o silencio detectado (No features).")
                return "ERROR_NOISE"

            # 3. Predicción
            label, confidence = self.knn.predict(features)
            
            self.logger.info(f"🗣️ Comando: '{label.upper()}' | Confianza: {confidence:.2f}")
            
            # Umbral de confianza
            if confidence < 0.5:
                self.logger.warning("Confianza baja, ignorando comando.")
                return "UNCERTAIN"

            return label

        except Exception as e:
            self.logger.error(f"❌ Error crítico en micrófono/predicción: {e}")
            return "ERROR_MIC"