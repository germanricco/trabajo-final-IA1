import os
import subprocess
import sys

# --- CONFIGURACIÓN DE AUDIO (CONSISTENCIA PARA KNN) ---
SAMPLE_RATE = "16000"   # 16 kHz
CHANNELS = "1"          # Mono
FORMAT = "S16_LE"       # 16-bit Signed
DURATION = "2"          # 2 Segundos
DATASET_DIR = "audio"   # Carpeta raíz donde se guardará todo

def obtener_siguiente_indice(directorio, palabra):
    """
    Analiza la carpeta para encontrar el último número utilizado.
    Si existen 'contar_001.wav' y 'contar_002.wav', devuelve 3.
    """
    if not os.path.exists(directorio):
        return 1
    
    archivos = os.listdir(directorio)
    indices = []
    
    for archivo in archivos:
        # Filtramos solo archivos que empiecen con la palabra y terminen en .wav
        if archivo.startswith(palabra + "_") and archivo.endswith(".wav"):
            try:
                # Extraemos el número entre la palabra y .wav (ej: "contar_003.wav" -> 3)
                parte_numero = archivo.replace(palabra + "_", "").replace(".wav", "")
                indices.append(int(parte_numero))
            except ValueError:
                continue # Ignorar archivos que no cumplan el formato exacto
                
    if not indices:
        return 1
    else:
        return max(indices) + 1

def grabar_audio(filepath):
    """
    Ejecuta el comando arecord desde Python usando subprocess.
    """
    comando = [
        "arecord",
        "-f", FORMAT,
        "-c", CHANNELS,
        "-r", SAMPLE_RATE,
        "-d", DURATION,
        "-t", "wav",
        # "-D", "plughw:1,0",  # <--- DESCOMENTAR Y AJUSTAR SI USAS UN MICRO ESPECÍFICO
        filepath
    ]
    
    print(f"🎤 Grabando por {DURATION} segundos...", end=" ", flush=True)
    
    # Ejecutamos el comando. check=True lanzará error si arecord falla.
    try:
        subprocess.run(comando, check=True, stderr=subprocess.DEVNULL)
        print("✅ Guardado.")
    except subprocess.CalledProcessError:
        print("\n❌ Error: 'arecord' falló. ¿Está conectado el micrófono?")
        sys.exit(1)

def main():
    print("--- 🎙️ GENERADOR DE DATASET DE AUDIO 🎙️ ---")
    
    # 1. Pedir la etiqueta (palabra) una sola vez al inicio
    palabra = input("¿Qué palabra vamos a grabar? (ej: contar): ").strip().lower()
    
    if not palabra:
        print("Debes escribir una palabra.")
        return

    # 2. Crear estructura de carpetas: ./dataset/contar/
    ruta_carpeta = os.path.join(DATASET_DIR, palabra)
    os.makedirs(ruta_carpeta, exist_ok=True)
    print(f"📂 Directorio de trabajo: {ruta_carpeta}")

    # 3. Bucle de grabación
    while True:
        # Calcular el índice actual (cada vez, por si borraste algo manualmente)
        indice = obtener_siguiente_indice(ruta_carpeta, palabra)
        nombre_archivo = f"{palabra}_{indice:03d}.wav"
        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)

        print(f"\n--- Preparando grabación #{indice} para '{palabra}' ---")
        input(f"Presiona [ENTER] para empezar a grabar '{nombre_archivo}'...")
        
        # Grabar
        grabar_audio(ruta_completa)
        
        # 4. Preguntar si continuar
        while True:
            respuesta = input("¿Grabar otra muestra? (s/n): ").lower().strip()
            if respuesta in ['s', 'si', 'y', 'yes', '']: # Enter vacío = Sí (para ir rápido)
                break # Sale del bucle de pregunta y vuelve a grabar
            elif respuesta in ['n', 'no']:
                print("\n👋 Sesión finalizada.")
                print(f"Total grabado en esta sesión: {ruta_carpeta}")
                return # Termina el programa
            else:
                print("Por favor, responde 's' o 'n'.")

if __name__ == "__main__":
    main()