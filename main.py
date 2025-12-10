import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os

# Importar arquitectura modular
from src.core.agent import HardwareAgent
from src.ui.panels import VisionPanel, ControlPanel, BayesianAnalysisPanel, StatusFooter, AudioFooter

class App:
    def __init__(self, root):
        # Configuración de la ventana principal
        self.root = root
        self.root.title("Sistema de Clasificacion Industrial v2.0")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 768)
        
        # Configuracion de estilo
        style = ttk.Style()
        style.theme_use("clam")

        # Backend: Instanciar Agente
        self.agent = HardwareAgent()

        # Obtener referencia segura al preprocesador de imagenes
        if self.agent.vision:
            shared_preprocessor = self.agent.vision.img_prep
        else:
            from src.vision.preprocessor import ImagePreprocessor
            shared_preprocessor = ImagePreprocessor()

        self.current_image_path = None
        
        # --- FRONTEND: Layout Principal (grid) ---
        # Fila 0: Contenido Principal (Se estira mucho)
        self.root.rowconfigure(0, weight=1)
        # Fila 1: Footer (Altura fija, no se estira verticalmente)
        self.root.rowconfigure(1, weight=0)
        
        # Columna 0: Zona vision (3/4 espacio)
        self.root.columnconfigure(0, weight=3) # Imagen
        # Columna 1: Sidebar controles (1/4 espacio)
        self.root.columnconfigure(1, weight=1) # Sidebar

        # ==========================================
        #        COLUMNA 0 (IZQUIERDA)
        # ==========================================

        # PANEL DE VISION (arriba)
        self.vision_container = tk.Frame(self.root, bg="#2c3e50")
        self.vision_container.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.vision_panel = VisionPanel(self.vision_container, preprocessor=shared_preprocessor)
        self.vision_panel.pack(fill="both", expand=True)

        # FOOTER DE ESTADO (Abajo)
        self.status_footer = StatusFooter(self.root)
        self.status_footer.grid(row=1, column=0, sticky="ew", padx=1, pady=(0, 1))

        # ==========================================
        #        COLUMNA 1 (DERECHA)
        # ==========================================

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
            on_reset=self.reset_batch
        )
        self.ctrl_panel.pack(side="top", fill="x")
        
        # Panel de Análisis Bayesiano
        self.stats_panel = BayesianAnalysisPanel(self.sidebar_container)
        self.stats_panel.pack(side="top", fill="both", expand=True)

        # Footer de Audio
        self.audio_footer = AudioFooter(self.root, on_listen_click=self.listen_voice_command)
        self.audio_footer.grid(row=1, column=1, sticky="ew", padx=1, pady=(0, 1))
        
        
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

        # 2. Dibujar cajas en la imagen
        self.vision_panel.draw_detections(detections) # Pasamos la lista directa de diccionarios
        
        # 3. Actualizar Gráficos y Estadísticas
        if estimation:
            self.stats_panel.update_view(estimation)
        
        # 4. Feedback en barra de estado
        count = result.get('count_in_image', 0)
        self.status_footer.set_text(f"✅ Análisis completado: {count} objetos añadidos.")


    def reset_batch(self):
        """Reinicia la memoria del estimador bayesiano."""
        if messagebox.askyesno("Reset", "¿Reiniciar lote?"):
            self.agent.reset_estimation()
            self.stats_panel.reset_view()
            self.status_footer.set_text("🔄 Lote reiniciado.")


    def run_training(self):
        """Entrenamiento en hilo secundario."""
        if not messagebox.askyesno("Entrenar Sistema", 
                                   "¿Iniciar re-entrenamiento?\nEsto optimizará los clusters K-Means."):
            return
            
        self.status_footer.set_text("🧠 Entrenando... (No cierre la ventana)")
        self.root.config(cursor="wait") # Poner cursor de reloj
        

        def _train_thread():
            try:
                acc = self.agent.train_vision_system()
                self.root.after(0, lambda: self._training_complete(acc))
            except Exception as e:
                self.root.after(0, lambda: self._training_error(str(e)))
            
        threading.Thread(target=_train_thread, daemon=True).start()
    

    def _training_complete(self, accuracy):
        self.root.config(cursor="")
        msg = f"Entrenamiento Exitoso.\nPrecisión del modelo: {accuracy:.2%}"
        self.status_footer.set_text(f"Modelo listo ({accuracy:.1%})")
        messagebox.showinfo("Éxito", msg)


    def _training_error(self, error_msg):
        self.root.config(cursor="")
        self.status_footer.set_text("Error en entrenamiento")
        messagebox.showerror("Error Crítico", f"Falló el entrenamiento:\n{error_msg}")


    def load_existing_model(self):
        """Carga modelo pre-entrenado."""
        if self.agent.load_trained_model():
            self.status_footer.set_text("📂 Modelo cargado desde disco.")
            messagebox.showinfo("Carga Exitosa", "El sistema de visión está listo.")
        else:
            self.status_footer.set_text("❌ Error cargando modelo.")
            messagebox.showerror("Error", "No se encontró 'vision_system.pkl'.")


    def listen_voice_command(self):
        """
        Placeholder para la futura lógica de audio.
        """
        # Aquí luego llamaremos a self.agent.listen()
        self.audio_footer.set_text("👂 Escuchando... (Próximamente)", is_active=True)
        self.status_footer.set_text("🎤 Interacción de voz iniciada...")
        
        # Simulamos delay de procesamiento
        self.root.after(2000, lambda: self._on_voice_processed())


    def _on_voice_processed(self):
        self.audio_footer.set_text("Comando no reconocido", is_active=False)
        self.status_footer.set_text("Listo")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()