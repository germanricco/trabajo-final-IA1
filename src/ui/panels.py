import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
            bbox_proc = item['bbox'] # (x, y, w, h)
            
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


class StatusFooter(tk.Frame):
    """
    Barra inferior izquierda. Muestra logs generales del sistema.
    Alineada con el panel de visión.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, bg="#2c3e50", height=40) # Altura fija
        self.pack_propagate(False) 
        
        # Icono decorativo
        tk.Label(self, text="ℹ️", bg="#2c3e50", fg="#ecf0f1", font=("Arial", 12)).pack(side="left", padx=(10, 5))
        
        # Texto de estado
        self.lbl_status = tk.Label(self, text="Sistema Listo", 
                                   bg="#2c3e50", fg="#bdc3c7", 
                                   font=("Consolas", 10), anchor="w")
        self.lbl_status.pack(side="left", fill="both", expand=True, padx=5)

    def set_text(self, text):
        self.lbl_status.config(text=text)


# --- 2. FOOTER DERECHO: INTERFAZ DE VOZ ---
class AudioFooter(tk.Frame):
    """
    Barra inferior derecha. Controles de micrófono y feedback de texto.
    Alineada con el panel de control.
    """
    def __init__(self, parent, on_listen_click=None, *args, **kwargs):
        super().__init__(parent, bg="#34495e", height=40) # Un tono ligeramente diferente para distinguir
        self.pack_propagate(False)
        self.on_listen_click = on_listen_click

        # Botón Micrófono (Estilo "Action Button")
        self.btn_mic = tk.Button(self, text="🎙️", command=self._on_click,
                                 bg="#e74c3c", fg="white", relief="flat", 
                                 font=("Arial", 12), cursor="hand2", bd=0, width=4)
        self.btn_mic.pack(side="right", fill="y", padx=0)

        # Separador visual vertical
        tk.Frame(self, bg="#7f8c8d", width=1).pack(side="right", fill="y")

        # Feedback de texto (Output del STT)
        self.lbl_feedback = tk.Label(self, text="Presione para hablar...", 
                                     bg="#34495e", fg="#ecf0f1", 
                                     font=("Segoe UI", 9, "italic"), anchor="e")
        self.lbl_feedback.pack(side="right", fill="both", expand=True, padx=10)

    def set_text(self, text, is_active=False):
        self.lbl_feedback.config(text=text)
        if is_active:
            self.lbl_feedback.config(fg="#f1c40f", font=("Segoe UI", 9, "bold")) # Amarillo y negrita
        else:
            self.lbl_feedback.config(fg="#ecf0f1", font=("Segoe UI", 9, "italic"))

    def _on_click(self):
        if self.on_listen_click:
            self.on_listen_click()


class ControlPanel(tk.Frame):
    """
    Panel lateral con botones de acción y gestión del flujo de trabajo.
    """
    def __init__(self, parent, on_load, on_predict, on_train, on_load_model, on_reset=None):
        super().__init__(parent, bg="white") 
        
        # --- SECCIÓN 1: OPERACIÓN ---
        tk.Label(self, text="OPERACIÓN", font=("Segoe UI", 9, "bold"), 
                 bg="white", fg="#95a5a6").pack(fill="x", padx=10, pady=(20, 5), anchor="w")
        
        # Botón Cargar
        self._make_btn("📂 Cargar Imagen", on_load, "#34495e").pack(fill="x", padx=10, pady=5)
        
        # Botón Predecir (Analizar)
        btn_predict = self._make_btn("🧠 ANALIZAR MUESTRA", on_predict, "#27ae60")
        btn_predict.config(font=("Segoe UI", 11, "bold"), pady=8) 
        btn_predict.pack(fill="x", padx=10, pady=5)

        # Botón Reset (Opcional)
        if on_reset:
            self._make_btn("🔄 Reiniciar Lote", on_reset, "#c0392b").pack(fill="x", padx=10, pady=5)
        
        # --- SEPARADOR ---
        tk.Frame(self, height=2, bg="#ecf0f1").pack(fill="x", padx=20, pady=15)

        # --- SECCIÓN 2: MANTENIMIENTO ---
        tk.Label(self, text="SISTEMA", font=("Segoe UI", 9, "bold"), 
                 bg="white", fg="#95a5a6").pack(fill="x", padx=10, pady=(5, 5), anchor="w")

        self._make_btn("💾 Cargar Modelo", on_load_model, "#7f8c8d").pack(fill="x", padx=10, pady=2)
        self._make_btn("⚙️ Re-Entrenar IA", on_train, "#d35400").pack(fill="x", padx=10, pady=5)
    

    def _make_btn(self, text, cmd, color):
        return tk.Button(self, text=text, command=cmd, bg=color, fg="white", 
                         font=("Segoe UI", 10), cursor="hand2", 
                         relief="flat", activebackground="#2c3e50", activeforeground="white")

    def set_status(self, text):
        self.lbl_status.config(text=text)

    def log_message(self, text):
        self.set_status(text)


class BayesianAnalysisPanel(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config(bg="#f0f0f0", padx=10, pady=10)

        # --- SECCIÓN 1: ESTADO DEL LOTE ---
        self.info_frame = tk.Frame(self, bg="#f0f0f0")
        self.info_frame.pack(fill="x", pady=(0, 10))

        # Título y Contador
        self.lbl_title = tk.Label(self.info_frame, text="📊 ANÁLISIS DE LOTE", 
                                  font=("Helvetica", 12, "bold"), bg="#f0f0f0", fg="#333")
        self.lbl_title.pack(anchor="w")

        self.lbl_counter = tk.Label(self.info_frame, text="Piezas acumuladas: 0 / 10", 
                                    font=("Helvetica", 10), bg="#f0f0f0", fg="#666")
        self.lbl_counter.pack(anchor="w")

        # Barra de Progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.info_frame, variable=self.progress_var, maximum=10)
        self.progress_bar.pack(fill="x", pady=5)

        # --- SECCIÓN 2: GRÁFICO DE PROBABILIDAD ---
        # Creamos una Figura de Matplotlib pequeña pero clara
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor('#f0f0f0') # Fondo igual al de la UI
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- SECCIÓN 3: RESULTADO FINAL ---
        self.result_frame = tk.Frame(self, bg="#e0e0e0", bd=2, relief="groove")
        self.result_frame.pack(fill="x", pady=(10, 0))
        
        self.lbl_result = tk.Label(self.result_frame, text="ESPERANDO DATOS...", 
                                   font=("Arial", 14, "bold"), bg="#e0e0e0", fg="#888")
        self.lbl_result.pack(pady=10)

        # Inicializar gráfico vacío
        self.update_chart({'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25})

    def update_view(self, estimation_result: dict):
        """
        Recibe el diccionario 'estimation_result' del Agente y actualiza toda la UI.
        """
        if not estimation_result:
            self.reset_view()
            return

        total_seen = estimation_result['total_seen']
        probs = estimation_result['all_probs']
        is_ready = estimation_result['is_ready']
        
        # 1. Actualizar Barra
        self.progress_var.set(min(total_seen, 10))
        self.lbl_counter.config(text=f"Piezas acumuladas: {total_seen} / 10")
        
        # 2. Actualizar Gráfico
        self.update_chart(probs)
        
        # 3. Actualizar Resultado Final
        if is_ready:
            top_box = estimation_result['top_box'][0]  # Letra de la caja (ej: 'A')
            conf = estimation_result['confidence'] * 100
            
            self.lbl_result.config(
                text=f"✅ CAJA DETECTADA: {top_box}\n(Confianza: {conf:.1f}%)",
                fg="#006400", # Verde oscuro
                bg="#ccffcc"
            )
            self.result_frame.config(bg="#ccffcc")
        else:
            self.lbl_result.config(text="ANALIZANDO...", fg="#d35400", bg="#ffeebb")
            self.result_frame.config(bg="#ffeebb")

    def update_chart(self, probs: dict):
        self.ax.clear()
        
        # Datos
        cajas = list(probs.keys())
        valores = [p * 100 for p in probs.values()] # A porcentaje
        colores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'] # Azul, Rojo, Verde, Violeta

        # Dibujar Barras
        bars = self.ax.bar(cajas, valores, color=colores, alpha=0.8)
        
        # Estética
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Probabilidad (%)", fontsize=8)
        self.ax.tick_params(axis='both', which='major', labelsize=8)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.2, right=0.95)

        # Etiquetas sobre las barras
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{height:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        self.canvas.draw()

    def reset_view(self):
        self.progress_var.set(0)
        self.lbl_counter.config(text="Piezas acumuladas: 0 / 10")
        self.update_chart({'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25})
        self.lbl_result.config(text="ESPERANDO DATOS...", fg="#888", bg="#e0e0e0")
        self.result_frame.config(bg="#e0e0e0")



class AudioPanel(tk.Frame):
    """
    Panel dedicado a la interacción por voz.
    Contiene botón de micrófono y área de feedback de texto.
    """
    def __init__(self, parent, on_listen_click=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config(bg="#f0f0f0", bd=1, relief="sunken", padx=10, pady=10)
        self.on_listen_click = on_listen_click

        # Título pequeño
        lbl_title = tk.Label(self, text="🎤 COMANDOS DE VOZ", 
                             font=("Segoe UI", 8, "bold"), bg="#f0f0f0", fg="#666")
        lbl_title.pack(anchor="w", pady=(0, 5))

        # Contenedor horizontal para botón + texto
        container = tk.Frame(self, bg="#f0f0f0")
        container.pack(fill="x")

        # Botón Micrófono (Simulado con texto/emoji por ahora, luego puedes poner un icono PNG)
        self.btn_mic = tk.Button(container, text="🎙️", font=("Arial", 16), 
                                 command=self._on_click,
                                 bg="#e74c3c", fg="white", # Rojo estilo 'Grabar'
                                 activebackground="#c0392b", activeforeground="white",
                                 relief="flat", width=3)
        self.btn_mic.pack(side="left", padx=(0, 10))

        # Barra de Texto (Output del STT)
        self.txt_output = tk.Entry(container, font=("Consolas", 10), bg="white", fg="#333", relief="flat")
        self.txt_output.insert(0, "Esperando comando...")
        self.txt_output.config(state="readonly") # Solo lectura para el usuario
        self.txt_output.pack(side="left", fill="x", expand=True, ipady=4)

    def _on_click(self):
        if self.on_listen_click:
            self.on_listen_click()
        else:
            # Feedback visual temporal
            self.set_text("Escuchando... (Simulación)")

    def set_text(self, text):
        self.txt_output.config(state="normal")
        self.txt_output.delete(0, tk.END)
        self.txt_output.insert(0, text)
        self.txt_output.config(state="readonly")