import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

# Importar arquitectura modular
from src.core.agent import HardwareAgent
from src.ui.panels import VisionPanel, ControlPanel

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Visión Artificial v1.0")
        self.root.geometry("1100x700")
        
        # 1. Backend: Instanciar Agente
        self.agent = HardwareAgent()

        if self.agent.vision:
            shared_preprocessor = self.agent.vision.img_prep
        else:
            # Fallback de seguridad por si falla la inicialización del agente
            # (Aunque idealmente el agente maneja sus propios errores)
            from src.vision.preprocessor import ImagePreprocessor
            shared_preprocessor = ImagePreprocessor()

        self.current_image_path = None
        
        # 2. Frontend: Layout
        # Panel Izquierdo (Imagen)
        self.vision_panel = VisionPanel(root, preprocessor=shared_preprocessor)
        self.vision_panel.pack(side="left", fill="both", expand=True)
        
        # Panel Derecho (Controles)
        self.ctrl_panel = ControlPanel(root, 
                                     on_load=self.load_image,
                                     on_predict=self.run_detection,
                                     on_train=self.run_training,
                                     on_load_model=self.load_existing_model)
        self.ctrl_panel.pack(side="right", fill="y")


    def load_image(self):
        """Abre explorador de archivos."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.current_image_path = file_path
            self.vision_panel.load_image(file_path)
            self.ctrl_panel.set_status(f"Cargado: {os.path.basename(file_path)}")


    def run_detection(self):
        """Ejecuta la visión artificial."""
        if not self.current_image_path:
            messagebox.showwarning("Atención", "Carga una imagen primero.")
            return
            
        self.ctrl_panel.set_status("Analizando...")
        self.root.update() # Forzar refresco UI
        
        # Llamada al Agente
        results = self.agent.detect_objects(self.current_image_path)
        
        if not results:
            self.ctrl_panel.set_status("No se detectaron objetos.")
            messagebox.showinfo("Info", "No encontré nada conocido en la imagen.")
            return
            
        # Dibujar resultados
        self.vision_panel.draw_detections(results)
        
        # Resumen
        counts = {}
        for r in results:
            lbl = r['label']
            counts[lbl] = counts.get(lbl, 0) + 1
        
        summary = ", ".join([f"{k}: {v}" for k,v in counts.items()])
        self.ctrl_panel.set_status(f"Detectado:\n{summary}")


    def run_training(self):
        """
        Ejecuta el entrenamiento en un hilo secundario para no congelar la UI.
        """

        if not messagebox.askyesno("Entrenar", "¿Seguro? Esto tomará unos segundos."):
            return
            
        self.ctrl_panel.set_status("Entrenando IA... Espere.")
        
        def _train_thread():
            accuracy = self.agent.train_vision_system()
            # Volver al hilo principal para mostrar mensaje
            self.root.after(0, lambda: self._training_complete(accuracy))
            
        threading.Thread(target=_train_thread, daemon=True).start()
    

    def load_existing_model(self):
        """Carga el .pkl existente sin entrenar."""
        if self.agent.load_trained_model():
            self.ctrl_panel.set_status("Modelo cargado. Listo para usar.")
            messagebox.showinfo("Éxito", "Modelo de visión cargado correctamente.")
        else:
            self.ctrl_panel.set_status("Error: No hay modelo.")
            messagebox.showerror("Error", "No se encontró un modelo entrenado ('vision_system.pkl').\nPor favor entrene el sistema primero.")


    def _training_complete(self, accuracy):
        msg = f"Entrenamiento finalizado.\nPrecisión: {accuracy:.2%}"
        self.ctrl_panel.set_status(msg)
        messagebox.showinfo("Éxito", msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()