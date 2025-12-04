import tkinter as tk
from PIL import Image, ImageTk

class VisionPanel(tk.Frame):
    """
    Panel encargado de mostrar la imagen y dibujar los resultados de la IA.
    Maneja el escalado de coordenadas entre la imagen real y la visualización.
    """
    def __init__(self, parent, preprocessor, width=600, height=800):
        super().__init__(parent, bg="#2c3e50", padx=10, pady=10)
        self.view_width = width
        self.view_height = height
        
        self.preprocessor = preprocessor

        # Canvas
        self.canvas = tk.Canvas(self, bg="black", width=width, height=height, highlightthickness=0)
        self.canvas.pack(expand=True)
        
        # Variables de estado
        self.current_image = None  # Objeto PIL original
        self.photo_ref = None      # Referencia para evitar Garbage Collection
        self.scale_factor_ui = 1.0    # Relación (View / Real)
        self.offset = (0, 0)       # Margen (x, y) si la imagen no llena el canvas

    def load_image(self, image_path):
        """Carga y muestra una imagen limpia."""
        self.current_image = Image.open(image_path)
        self._render_image()

    def _render_image(self):
        """
        Redimensiona la imagen para encajar en el canvas (Letterbox) y la dibuja.
        """
        if not self.current_image: return

        # 1. Calcular escala para "Fit" (encajar sin deformar)
        w_orig, h_orig = self.current_image.size
        
        # Factor de escala (minimo para que entre todo)
        scale_w = self.view_width / w_orig
        scale_h = self.view_height / h_orig
        self.scale_factor_ui = min(scale_w, scale_h)
        
        # 2. Nuevas dimensiones
        new_w = int(w_orig * self.scale_factor_ui)
        new_h = int(h_orig * self.scale_factor_ui)
        
        # 3. Resize visual
        resized_img = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo_ref = ImageTk.PhotoImage(resized_img)
        
        # 4. Calcular posición centrada
        self.offset_x = (self.view_width - new_w) // 2
        self.offset_y = (self.view_height - new_h) // 2
        
        # 5. Dibujar
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo_ref, anchor="nw")

    def draw_detections(self, detections):
        """
        Dibuja los Bounding Boxes sobre la imagen actual.
        """
        print(f"🖌️ UI: Intentando dibujar {len(detections)} detecciones...")
        
        if not self.current_image: 
            print("⚠️ UI: No hay imagen cargada para dibujar encima.")
            return
        
        self.canvas.delete("overlay")

        w_real, h_real = self.current_image.size

        # Colores por clase
        colors = {
            "tornillos": "#3498db", # Azul
            "clavos": "#e74c3c",    # Rojo
            "arandelas": "#f1c40f", # Amarillo
            "tuercas": "#9b59b6",   # Violeta
            "desconocido": "white"
        }

        for i, item in enumerate(detections):
            label = item['label']
            bbox_proc = item['bbox'] # (x, y, w, h) en 800x600
            
            # 1. Escalar de vuelta a original
            try:
                # Recuperar coords reales
                x_real, y_real, w_real_box, h_real_box = self.preprocessor.scale_bbox_back(bbox_proc, (h_real, w_real))
                
                # 2. Escalar a pantalla
                x1 = (x_real * self.scale_factor_ui) + self.offset_x
                y1 = (y_real * self.scale_factor_ui) + self.offset_y
                x2 = ((x_real + w_real_box) * self.scale_factor_ui) + self.offset_x
                y2 = ((y_real + h_real_box) * self.scale_factor_ui) + self.offset_y
                
                print(f"   -> Obj {i} ({label}): BBoxOriginal={bbox_proc} -> Canvas=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

                color = colors.get(label, "white")
                
                # Rectángulo (Tag 'overlay')
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3, tags="overlay")
                
                # Texto
                self.canvas.create_text(x1, y1-15, text=label.upper(), fill=color, 
                                      font=("Arial", 10, "bold"), anchor="sw", tags="overlay")
                                      
            except Exception as e:
                print(f"❌ Error dibujando objeto {i}: {e}")
        
        # Forzar actualización visual
        self.canvas.update()
class ControlPanel(tk.Frame):
    """Panel lateral con botones de acción."""
    def __init__(self, parent, on_load, on_predict, on_train, on_load_model):
        super().__init__(parent, bg="white", width=200)
        self.pack_propagate(False) # Mantener ancho fijo
        
        tk.Label(self, text="GESTIÓN IA", font=("Arial", 10, "bold"), bg="white", fg="gray").pack(pady=(20, 5))
        
        # Botones
        self._make_btn("📂 Cargar Imagen", on_load, "#34495e").pack(fill="x", padx=10, pady=5)
        self._make_btn("🧠 Detectar Objetos", on_predict, "#27ae60").pack(fill="x", padx=10, pady=5)
        
        tk.Frame(self, height=20, bg="white").pack() # Espaciador
        self._make_btn("💾 Cargar Modelo", on_load_model, "#7f8c8d").pack(fill="x", padx=10, pady=2) # Nuevo
        self._make_btn("⚙️ Re-Entrenar IA", on_train, "#e67e22").pack(fill="x", padx=10, pady=5)
        
        # Log simple
        self.lbl_status = tk.Label(self, text="Listo.", bg="white", fg="gray", wraplength=180)
        self.lbl_status.pack(side="bottom", pady=20)

    def _make_btn(self, text, cmd, color):
        return tk.Button(self, text=text, command=cmd, bg=color, fg="white", 
                         font=("Arial", 10, "bold"), height=2, cursor="hand2", relief="flat")

    def set_status(self, text):
        self.lbl_status.config(text=text)