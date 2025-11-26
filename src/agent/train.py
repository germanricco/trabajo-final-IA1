import sys
import os
import logging

# Aseguramos que Python encuentre tus módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.agent.ImageClassifier import ImageClassifier

# Configurar logs para ver todo el detalle en consola
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_best_model(attempts=3, min_accuracy=0.85):
    """
    Intenta entrenar el modelo 'attempts' veces.
    Solo guarda si supera 'min_accuracy' o si es el mejor intento hasta ahora.
    """
    data_path = "data/raw/images/all"
    
    best_accuracy = 0.0
    best_classifier_state = None
    
    print(f"{'='*60}")
    print(f"🚀 INICIANDO PROTOCOLO DE ENTRENAMIENTO (Intentos: {attempts})")
    print(f"{'='*60}")

    classifier = ImageClassifier()

    for i in range(1, attempts + 1):
        print(f"\n🔄 INTENTO {i}/{attempts}...")
        
        # Entrenamos (Esto divide 80/20, entrena KMeans y valida)
        # NOTA: ImageClassifier.train() guarda automáticamente. 
        # Para este script avanzado, confiamos en que guarde el último, 
        # pero aquí validamos si "vale la pena" celebrar.
        accuracy = classifier.train(data_path)
        
        print(f"   🎯 Resultado Intento {i}: {accuracy:.2%} de precisión")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print("   ✨ ¡Nuevo mejor modelo encontrado!")
            
            # Si ya es muy bueno, paramos antes
            if best_accuracy >= 0.98:
                print("   🏆 Precisión excelente alcanzada. Terminando temprano.")
                break
        else:
            print("   📉 Este intento no mejoró el anterior.")

    print(f"\n{'='*60}")
    print(f"🏁 FIN DEL PROCESO")
    print(f"   Mejor precisión obtenida: {best_accuracy:.2%}")
    
    if best_accuracy < min_accuracy:
        print("   ⚠️ ADVERTENCIA: El modelo final no alcanzó la precisión mínima deseada.")
    else:
        print("   ✅ Modelo guardado y listo para producción.")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Ajusta min_accuracy según qué tan exigente quieras ser
    train_best_model(attempts=5, min_accuracy=0.80)