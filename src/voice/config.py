from dataclasses import dataclass, field
from typing import Tuple, Dict

@dataclass
class VoiceConfig:
    """
    Contenedor centralizado de hiperparámetros para el sistema de voz.
    Permite versionar configuraciones y experimentar rápido.
    """

    # --- Preprocesamiento ---
    sample_rate: int = 16000
    top_db: int = 20
    pre_emphasis_coef: float = 0.97
    lowcut: float = 200.0
    highcut: float = 5500.0
    filter_order: int = 5
    n_mfcc: int = 13

    duration: int = 2
    
    # --- Modelo (KNN) ---
    k_neighbors: int = 5