from dataclasses import dataclass, field
from typing import Tuple, Dict

@dataclass
class VisionConfig:
    """
    Contenedor centralizado de hiperparámetros para el sistema de visión.
    Permite versionar configuraciones y experimentar rápido.
    """
    
    # --- Preprocesamiento ---
    target_size: Tuple[int, int] = (960, 1280)
    gamma: float = 1.7
    d_bFilter: int = 8
    binarization_block_size: int = 27
    binarization_C: int = -4
    open_kernel_size: Tuple[int, int] = (3, 3)
    close_kernel_size: Tuple[int, int] = (3, 3)
    clear_border_margin: int = 5
    
    # --- Segmentación ---
    min_area: int = 100
    merge_distance: int = 15
    
    # --- Modelo (K-Means) ---
    n_clusters: int = 4
    n_init: int = 50
    
    # --- Pesos de Características ---
    #? Usamos default_factory para evitar problemas con mutables en dataclasses
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'radius_variance': 7.0,
        'circle_ratio': 16.0,
        'hole_confidence': 2.0,
        'aspect_ratio': 3.0,
        'solidity': 1.0,
    })

    # --- Método helper para cargar desde JSON (Opcional pero recomendado) ---
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Permite cargar configuración desde un archivo externo (JSON/YAML)"""
        # Filtra claves que no pertenezcan a la clase para evitar errores
        valid_keys = {k for k in config_dict if k in cls.__annotations__}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)