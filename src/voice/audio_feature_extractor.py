import numpy as np
import librosa
from scipy.signal import butter, lfilter
import logging

class AudioFeatureExtractor:
    def __init__(self,
                 sr=16000,
                 top_db=20,
                 pre_emphasis_coef=0.97,
                 lowcut=200,
                 highcut=5500,
                 filter_order=5,
                 n_mfcc=13):
        """
        Inicializa el extractor de caracteristicas de audio.

        Args:
            * sr (int): Sample Rate.
            * top_db (int): Umbral en decibelios para eliminar silencios.
            * lowcut (float): Frecuencia de corte baja en Hz.
            * highcut (float): Frecuencia de corte alta en Hz.
            * n_mfcc (int): Número de coeficientes MFCC a extraer (13 es estandard).
            * pre_emphasis_coef (float): Coeficiente para el filtro de pre-énfasis.
        """
        self.sr = sr
        self.top_db = top_db
        self.pre_emphasis_coef = pre_emphasis_coef 
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.n_mfcc = n_mfcc

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AudioFeatureExtractor inicializado correctamente.")
        

    
    def extract_features(self, audio_input):
        """
        Procesa el audio y extrae un vector de caracteristicas completo.

        Args:
            * audio_input (str): Ruta al archivo de audio (str) o señal de audio (np.ndarray).

        Returns:
            * np.ndarray: Vector de características extraídas.
        """

        try:
            if isinstance(audio_input, str):
                # Cargar desde archivo, forzando el SR de 16000
                signal, _ = librosa.load(audio_input, sr=self.sr)
            else:
                # Señal de audio ya cargada
                signal = audio_input
            
            # Eliminar silencios al inicio y final
            signal = self._remove_silence(signal, top_db=self.top_db)

            # Pre-enfasis (balancear espectro de frecuencias)
            signal = self._pre_emphasis(signal, emphasis_coef=self.pre_emphasis_coef)

            # Aplicar el filtro pasa banda
            signal = self._bandpass_filter(signal, lowcut=self.lowcut, highcut=self.highcut, order=self.filter_order)
            
            # Normalizar la señal
            signal = self._normalize_audio(signal)

            #! Extracción de características
            # MFCC: La "forma" del tracto vocal
            mfcc = librosa.feature.mfcc(y=signal, sr=self.sr, n_mfcc=self.n_mfcc)
            # ZCR: Tasa de cruce por cero (ayuda a distinguir ruidos sibilantes 's' vs vocales)
            zcr = librosa.feature.zero_crossing_rate(y=signal)
            # Energía (RMS)
            energy = librosa.feature.rms(y=signal)

            # Aplicar (Pooling).
            # Tomamos el promedio de cada característica.
            mfcc_mean = np.mean(mfcc, axis=1)
            zcr_mean = np.mean(zcr)
            energy_mean = np.mean(energy)

            # Concatenar todas las características en un solo vector de 1D
            features = np.concatenate((mfcc_mean, [zcr_mean, energy_mean]))
            return features

        except Exception as e:
            self.logger.error(f"Error extrayendo características de audio: {e}")
            return None
        

    def _pre_emphasis(self, y, emphasis_coef=0.97):
        """
        Aplica un filtro de pre-énfasis a la señal de audio.
        Realiza frecuencias altas, compensando la caída natural en esas frecuencias.
        """
        return librosa.effects.preemphasis(y, coef=emphasis_coef)


    def _bandpass_filter(self, y, lowcut=200, highcut=5500, order=5):
        """
        Filtro Butterworth pasa banda.

        Args:
            * y (np.ndarray): Señal de audio.
            * lowcut (float): Frecuencia de corte baja en Hz.
            * highcut (float): Frecuencia de corte alta en Hz.
            * order (int): Orden del filtro.

        Returns:
            * np.ndarray: Señal de audio filtrada.
        """
        nyquist = 0.5 * self.sr

        # Highcut no puede ser mayor que Nyquist
        if highcut >= nyquist:
            highcut = nyquist - 100

        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        y_filtered = lfilter(b, a, y)
        return y_filtered


    def _remove_silence(self, y, top_db=20):
        """
        Elimina los silencios al inicio y final de la señal de audio.

        Args:
            * y (np.ndarray): Señal de audio.
            * top_db (int): Umbral en decibelios para considerar una parte como silencio

        Returns:
            * np.ndarray: Señal de audio sin silencios al inicio y final.
        """
        non_silent_intervals = librosa.effects.split(y, top_db=top_db)

        # Si todos son silencios, retornar la señal original
        if len(non_silent_intervals) == 0:
            return y
        
        y_trimmed = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        return y_trimmed


    def _normalize_audio(self, y):  
        """
        Normalizacion Global (Peak Normalization).
        Escala la señal para que su valor absoluto máximo sea 1.0.
        """
        max_abs = np.max(np.abs(y))
        if max_abs > 0:
            return y / max_abs
        return y