import os
import sys
from pathlib import Path

# 1. CÁLCULO ROBUSTO DEL ROOT_PATH
# Usamos __file__ en lugar de cwd() para que funcione sin importar desde dónde llames al script en la terminal
# Asumimos que este script está en: /tuproyecto/src/utils/train_voice_tool.py
CURRENT_FILE = Path(__file__).resolve()
ROOT_PATH = CURRENT_FILE.parent.parent.parent  # Subimos 3 niveles: utils -> src -> root

print(f"🔍 ROOT_PATH calculado: {ROOT_PATH}")

# Agregamos al path del sistema para poder importar los módulos
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

# Importación del Agente (ahora que sys.path es correcto)
from src.core.agent import HardwareAgent

def main():
    # 2. DEFINIR RUTAS ABSOLUTAS
    # Usamos el operador '/' de pathlib que es más limpio que os.path.join
    dataset_absolute_path = ROOT_PATH / "data" / "raw" / "audio"
    models_absolute_path = ROOT_PATH / "models"

    print(f"📂 Buscando dataset en: {dataset_absolute_path}")
    
    # Validación previa
    if not dataset_absolute_path.exists():
        print(f"❌ Error CRÍTICO: No existe la carpeta del dataset.")
        print(f"   Esperaba encontrarla en: {dataset_absolute_path}")
        return

    # 3. INICIALIZAR AGENTE CON RUTA DE MODELOS
    # Le pasamos models_dir explícitamente para que guarde el .pkl en el root, no en src/utils
    print("🤖 Inicializando HardwareAgent...")
    agent = HardwareAgent(models_dir=str(models_absolute_path))

    # --- PASO 1: ENTRENAMIENTO ---
    print("\n🧠 --- INICIANDO ENTRENAMIENTO DE VOZ ---")
    
    # Pasamos la ruta convertida a string (por compatibilidad con os.path dentro del agente)
    success = agent.train_voice_system(data_path=str(dataset_absolute_path))

    if success:
        print(f"✅ Entrenamiento exitoso.")
        print(f"   Modelo guardado en: {models_absolute_path / 'voice_model.pkl'}")
    else:
        print("❌ Falló el entrenamiento. Revisa los logs anteriores.")
        return

    # --- PASO 2: PRUEBA DE MICRÓFONO ---
    print("\n🎤 --- PRUEBA DE RECONOCIMIENTO EN VIVO ---")
    print("El sistema escuchará por 2 segundos.")
    print("Comandos válidos esperados: 'contar', 'proporcion', 'salir'")
    input("Presiona ENTER y habla inmediatamente >> ")

    # Escuchar
    try:
        command = agent.listen_command()
        
        print("\n" + "="*40)
        print(f"🗣️  RESULTADO: {command.upper()}")
        print("="*40)
        
        if command in ["contar", "proporcion", "salir"]:
            print("🎉 ¡Correcto! Comando válido reconocido.")
        elif "ERROR" in command:
            print("⚠️ Hubo un error técnico (Microfono o Modelo).")
        else:
            print("🤔 Se escuchó algo, pero no es un comando seguro (o ruido).")
            
    except KeyboardInterrupt:
        print("\n🛑 Prueba cancelada por el usuario.")

if __name__ == "__main__":
    main()