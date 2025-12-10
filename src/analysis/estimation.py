import numpy as np
from typing import Dict, List

class BoxEstimator:
    def __init__(self):
        # 1. Definimos los perfiles de las cajas (Hipótesis)
        # Convertimos las cantidades absolutas a probabilidades (normalizamos).
        # Agregamos un epsilon (1e-6) para evitar log(0) si la visión falla y ve algo que "no debería" estar.
        
        self.box_definitions = {
            'A': {'tornillos': 250, 'clavos': 250, 'arandelas': 250, 'tuercas': 250},
            'B': {'tornillos': 150, 'clavos': 300, 'arandelas': 300, 'tuercas': 250},
            'C': {'tornillos': 250, 'clavos': 350, 'arandelas': 250, 'tuercas': 150},
            'D': {'tornillos': 500, 'clavos': 500, 'arandelas': 0,   'tuercas': 0}
        }
        
        self.classes = ['arandelas', 'clavos', 'tornillos', 'tuercas']
        self.box_probs = self._calculate_box_probabilities()
        
        # Priors: Al inicio, todas las cajas son igualmente probables (25%)
        self.priors = {box: 0.25 for box in self.box_definitions}
        
        # Estado acumulado
        self.total_observed = 0
        self.evidence_history = {c: 0 for c in self.classes}

    def _calculate_box_probabilities(self):
        """Convierte conteos absolutos a probabilidades logarítmicas para estabilidad numérica."""
        probs = {}
        epsilon = 0.01 # 1% de margen de error para robustez ante falsos positivos
        
        for box, counts in self.box_definitions.items():
            total_items = sum(counts.values())
            box_p = {}
            for item in self.classes:
                # Probabilidad base: (Cantidad + epsilon) / Total
                # Esto asegura que Box D tenga una probabilidad muy baja (pero no cero) de tener arandelas
                p = (counts.get(item, 0) + epsilon) / (total_items + (epsilon * 4))
                box_p[item] = np.log(p) # Trabajamos con Logaritmos para sumar en vez de multiplicar
            probs[box] = box_p
        return probs

    def update(self, detected_counts: Dict[str, int]):
        """
        Actualiza las probabilidades de las cajas basándose en nuevos conteos.
        Usa Teorema de Bayes: P(Caja|Data) ∝ P(Data|Caja) * P(Caja)
        """
        self.total_observed += sum(detected_counts.values())
        
        # Actualizamos historial
        for k, v in detected_counts.items():
            if k in self.evidence_history:
                self.evidence_history[k] += v

        # Actualización Bayesiana
        new_priors = {}
        log_likelihoods = {}

        # 1. Calculamos el "score" de cada caja para los datos nuevos
        for box, log_probs in self.box_probs.items():
            # Empezamos con la probabilidad previa (en log)
            current_log_prob = np.log(self.priors[box] + 1e-10)
            
            # Sumamos la verosimilitud de cada objeto detectado
            for item, count in detected_counts.items():
                if item in log_probs:
                    current_log_prob += log_probs[item] * count
            
            log_likelihoods[box] = current_log_prob

        # 2. "Truco del Log-Sum-Exp" para normalizar y volver a probabilidades normales (0-1)
        # Esto evita problemas numéricos con números muy pequeños
        max_log = max(log_likelihoods.values())
        sum_exp = sum(np.exp(val - max_log) for val in log_likelihoods.values())
        
        for box in self.box_definitions:
            # P(Caja) normalizada
            new_priors[box] = np.exp(log_likelihoods[box] - max_log) / sum_exp

        self.priors = new_priors

    def get_prediction(self):
        """Retorna la Cajas más probable y su confianza, o None si no hay suficientes datos."""
        # Ordenamos las cajas por probabilidad descendente
        sorted_boxes = sorted(self.priors.items(), key=lambda x: x[1], reverse=True)
        top_box, confidence = sorted_boxes[0]
        
        return {
            "top_box": top_box,           # "A", "B", "C" o "D"
            "confidence": confidence,     # 0.0 a 1.0 (ej: 0.95)
            "all_probs": self.priors,     # {'A': 0.1, 'B': 0.8...}
            "total_seen": self.total_observed,
            "is_ready": self.total_observed >= 10 # Flag para la UI
        }

    def reset(self):
        self.priors = {box: 0.25 for box in self.box_definitions}
        self.total_observed = 0
        self.evidence_history = {c: 0 for c in self.classes}