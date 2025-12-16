import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from pathlib import Path
import logging

# Importar arquitectura modular
from src.core.agent import HardwareAgent
from src.ui.panels import VisionPanel
from src.ui.panels import ControlPanel
from src.ui.panels import BayesianAnalysisPanel
from src.ui.panels import StatusFooter
from src.ui.panels import AudioFooter

class App:
    def __init__(self, root):
        # Configuración de la ventana principal
        self.root = root
        self.root.title("Sistema de Clasificacion Industrial v3.2")
        self.root.geometry("1440x900")
        self.root.minsize(1024, 768)
        
        # Configuracion de estilo
        style = ttk.Style()
        style.theme_use("clam")

        self.logger = logging.getLogger("MainAPP")
        self.root_path = Path(__file__).resolve().parent

        # Backend: Instanciar Agente
        self.agent = HardwareAgent()

        # Obtener referencia segura al preprocesador de imagenes (visualizacion)
        if self.agent.vision:
            shared_preprocessor = self.agent.vision.img_prep
        else:
            from src.vision.preprocessor import ImagePreprocessor
            shared_preprocessor = ImagePreprocessor()

        self.current_image_path = None
        
        # FRONTEND: Layout Principal
        self.root.rowconfigure(0, weight=1) 
        self.root.rowconfigure(1, weight=0)
        self.root.columnconfigure(0, weight=3)  # Vision Panel
        self.root.columnconfigure(1, weight=1)  # Control Panel


        # ====== COLUMNA 0 (IZQUIERDA) ======

        # Panel de Vision
        self.vision_container = tk.Frame(self.root, bg="#2c3e50")
        self.vision_container.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.vision_panel = VisionPanel(self.vision_container, preprocessor=shared_preprocessor)
        self.vision_panel.pack(fill="both", expand=True)

        # Footer de Estado
        self.status_footer = StatusFooter(self.root)
        self.status_footer.grid(row=1, column=0, sticky="ew", padx=1, pady=(0, 1))

        # ====== COLUMNA 1 (DERECHA) ======

        # Contenedor de Sidebar de Controles
        self.sidebar_container = tk.Frame(self.root, bg="#f0f0f0")
        self.sidebar_container.grid(row=0, column=1, sticky="nsew", padx=1, pady=1)
        
        # Panel de Controles
        self.ctrl_panel = ControlPanel(
            self.sidebar_container, 
            on_load=self.load_image,
            on_predict=self.run_analysis,
            on_train=self.run_training,
            on_load_model=self.load_existing_model,
            on_reset=self.reset_batch,
            on_voice_train=self.run_voice_training
        )
        self.ctrl_panel.pack(side="top", fill="x")
        
        # Panel de Análisis Bayesiano
        self.stats_panel = BayesianAnalysisPanel(self.sidebar_container)
        self.stats_panel.pack(side="top", fill="both", expand=True)

        # Footer de Audio
        self.audio_footer = AudioFooter(self.root, on_listen_click=self.listen_voice_command)
        self.audio_footer.grid(row=1, column=1, sticky="ew", padx=1, pady=(0, 1))
        

    # =========================================================================
    # CLASIFICACIÓN DE IMÁGENES & BAYES
    # =========================================================================
    def load_image(self):
        """Abre explorador de archivos."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.current_image_path = file_path
            self.vision_panel.load_image(file_path)
            self.status_footer.set_text(f"Imagen cargada: {os.path.basename(file_path)}")


    def run_analysis(self):
        """
        Ejecuta el flujo completo de análisis bayesiano.
        Vision -> Bayes -> UI
        """
        if not self.current_image_path:
            messagebox.showwarning("Atención", "Carga una imagen primero.")
            return
            
        self.status_footer.set_text("🔍 Procesando imagen...")
        self.root.update()
        
        # Llamada al Agente (Orquestador)
        result = self.agent.process_sample_for_estimation(self.current_image_path)
        
        detections = result.get("detections", [])
        estimation = result.get("estimation_result")

        if not detections:
            self.status_footer.set_text("⚠️ No se detectaron objetos.")
            messagebox.showinfo("Info", "La imagen parece estar vacía o los objetos no son reconocibles.")
            return

        # Actualizar UI
        self.vision_panel.draw_detections(detections) # Pasamos la lista directa de diccionarios
        
        if estimation:
            self.stats_panel.update_view(estimation)
        
        # Feedback en barra de estado
        count = result.get('count_in_image', 0)
        self.status_footer.set_text(f"✅ Análisis completado: {count} objetos.")


    def reset_batch(self):
        """
        Reinicia la memoria del estimador bayesiano y limpia la UI.
        """
        if messagebox.askyesno("Reset", "¿Reiniciar lote?"):
            # Limpiar referencia al archivo actual
            self.current_image_path = None
            # Reiniciar logica interna de estimador bayesiano
            self.agent.reset_estimation()

            # Reiniciar paneles visuales
            self.stats_panel.reset_view()
            self.vision_panel.clear_image()

            # Feedback en barra de estado
            self.status_footer.set_text("🔄 Lote reiniciado.")


    def run_training(self):
        """Entrenamiento en hilo secundario."""
        if not messagebox.askyesno("Entrenar Sistema", 
                                   "¿Iniciar re-entrenamiento?\nEsto optimizará los clusters K-Means."):
            return
            
        self.status_footer.set_text("🧠 Entrenando... (No cierre la ventana)")
        self.root.config(cursor="watch") # Poner cursor de reloj
        

        def _train_thread():
            try:
                acc = self.agent.train_vision_system()
                self.root.after(0, lambda: self._training_complete(acc))
            except Exception as e:
                self.root.after(0, lambda: self._training_error(str(e)))
            
        threading.Thread(target=_train_thread, daemon=True).start()
    

    def _training_complete(self, accuracy):
        self.root.config(cursor="")
        self.status_footer.set_text(f"Modelo listo ({accuracy:.1%})")
        messagebox.showinfo("Éxito", f"Precisión del modelo: {accuracy:.2%}")


    def _training_error(self, error_msg):
        self.root.config(cursor="")
        self.status_footer.set_text("Error en entrenamiento")
        messagebox.showerror("Error Crítico", error_msg)


    def load_existing_model(self):
        """Carga modelo pre-entrenado."""
        if self.agent.load_trained_model():
            self.status_footer.set_text("📂 Modelo cargado desde disco.")
            messagebox.showinfo("Exito", "Modelo cargado correctamente.")
        else:
            self.status_footer.set_text("❌ Error cargando modelo.")
            messagebox.showerror("Error", "No se encontró el archivo del modelo.")


    # =========================================================================
    # RECONOCIMIENTO DE VOZ (INTEGRACIÓN COMPLETA)
    # =========================================================================

    def listen_voice_command(self):
        """
        Inicia el proceso de escucha en un hilo separado (para no congelar la UI).
        """
        # Feedback Visual Inmediato
        self.audio_footer.set_text("👂 Escuchando... (Hable ahora)", is_active=True)
        self.status_footer.set_text("🎤 Micrófono activo...")
        self.root.config(cursor="watch")
        
        # Lanzar Hilo
        threading.Thread(target=self._thread_voice_listener, daemon=True).start()


    def _thread_voice_listener(self):
        """Hilo secundario que espera al micrófono."""
        # Llamada bloqueante al agente (2 segundos aprox)
        command = self.agent.listen_command()
        
        # Volver al hilo principal
        self.root.after(0, lambda: self._process_voice_result(command))


    def _process_voice_result(self, command):
        """Se ejecuta en el hilo principal con el resultado."""
        # Restaurar UI
        self.root.config(cursor="")
        self.audio_footer.set_text("Click para hablar", is_active=False)
        
        # Manejo de Errores
        if "ERROR" in command:
            self.status_footer.set_text("❌ Error de Audio")
            messagebox.showwarning("Voz", "No se pudo reconocer el audio o hubo un error de hardware.")
            return

        if command == "UNCERTAIN":
            self.status_footer.set_text("⚠️ Comando incierto")
            return

        # Éxito
        self.status_footer.set_text(f"🗣️ Comando: {command.upper()}")
        
        # --- DISPATCHER DE COMANDOS ---
        if command == "salir":
            self._voice_action_exit()
            
        elif command == "contar":
            self._voice_action_count()
            
        elif command == "proporcion":
            self._voice_action_proportion()
            
        else:
            self.status_footer.set_text(f"Comando '{command}' desconocido")


    def run_voice_training(self):
        """
        Ejecuta el entrenamiento del modelo de voz.
        Determina la ruta absoluta de los datos y maneja el feedback UI.
        """
        # 1. Confirmación de seguridad
        if not messagebox.askyesno("Entrenar Voz", "¿Desea re-entrenar el modelo de voz?\nEsto puede tomar unos segundos."):
            return

        # 2. Definición Robusta de la Ruta
        # Asumimos estructura: raiz_proyecto/data/raw/audio
        data_path = self.root_path / "data" / "raw" / "audio"

        if not data_path.exists():
            self.logger.error(f"Ruta de dataset no encontrada: {data_path}")
            messagebox.showerror("Error", f"No se encuentra la carpeta de audios:\n{data_path}")
            return

        # 3. Feedback visual "Cargando"
        self.status_footer.set_text("🎙️ Entrenando modelo de voz... Por favor espere.")
        self.root.update_idletasks() # Fuerza a la UI a actualizarse antes de congelarse por el proceso

        try:
            # 4. Ejecución
            success = self.agent.train_voice_system(data_path=str(data_path))

            # 5. Resultado
            if success:
                self.status_footer.set_text("✅ Voz entrenada correctamente.")
                self.logger.info("Entrenamiento de voz finalizado con éxito.")
                messagebox.showinfo("Éxito", "El modelo de reconocimiento de voz ha sido actualizado.")
            else:
                self.status_footer.set_text("❌ Error en entrenamiento de voz.")
                messagebox.showerror("Error", "Ocurrió un error durante el entrenamiento.\nRevise la consola para más detalles.")

        except Exception as e:
            self.logger.exception("Excepción no controlada durante el entrenamiento de voz")
            self.status_footer.set_text("❌ Error crítico.")
            messagebox.showerror("Error Crítico", f"Ocurrió una excepción:\n{str(e)}")

    # --- ACCIONES ESPECÍFICAS DE VOZ ---
    def _voice_action_exit(self):
        if messagebox.askokcancel("Salir", "¿Confirmar salida por voz?"):
            self.root.quit()

    def _voice_action_count(self):
        """
        Muestra el reporte generado por el Agente sobre la última imagen.
        """
        report = self.agent.get_count_report()
        # Mostramos en un popup y en el log
        messagebox.showinfo("Reporte: Contar", report)
        print(f"Agente: {report}")

    def _voice_action_proportion(self):
        """
        Muestra el reporte generado por el Agente sobre la inferencia Bayesiana.
        """
        report = self.agent.get_proportion_report()
        messagebox.showinfo("Reporte: Proporción", report)
        print(f"Agente: {report}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()