import os
import csv
import sys
import trimesh
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class VaseLabelingTool:
    def __init__(self, root, models_dir, metadata_file):
        self.root = root
        self.root.title("Herramienta de Etiquetado de Floreros 3D")
        self.root.geometry("1200x800")

        self.models_dir = models_dir
        self.metadata_file = metadata_file

        # Cargar o crear archivo de metadatos
        self.load_metadata()

        # Lista de tipos de floreros
        self.vase_types = ["Estándar", "Con cuello", "Cilíndrico", "Esférico", "Asimétrico", "Artístico"]
        self.vase_styles = ["Liso", "Geométrico", "Orgánico", "Floral", "Texturizado", "Módular"]
        self.complexity_levels = ["1 - Simple", "2 - Media", "3 - Compleja", "4 - Muy compleja"]

        # Variables de etiquetado
        self.current_model_index = 0
        self.current_model = None
        self.model_files = self.get_model_files()

        # Crear interfaz
        self.create_widgets()

        # Cargar primer modelo
        if self.model_files:
            self.load_model(0)

    def get_model_files(self):
        """Obtiene lista de archivos de modelos 3D en el directorio"""
        model_files = []
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.lower().endswith(('.stl', '.obj', '.ply')):
                    model_files.append(os.path.join(root, file))
        return model_files

    def load_metadata(self):
        """Carga o crea archivo de metadatos"""
        self.metadata = {}

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata[row['filename']] = row
        else:
            # Crear archivo con encabezados
            with open(self.metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'filename', 'type', 'style', 'complexity',
                              'wall_thickness', 'needs_supports', 'print_time_estimate',
                              'filament_usage', 'tags', 'notes']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def create_widgets(self):
        """Crea la interfaz de usuario"""
        # Panel principal con dos secciones
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel izquierdo: visualización del modelo
        self.vis_frame = ttk.Frame(main_frame)
        main_frame.add(self.vis_frame, weight=2)

        # Configurar visualización del modelo
        self.fig = plt.figure(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Panel derecho: Formulario de etiquetado
        form_frame = ttk.Frame(main_frame)
        main_frame.add(form_frame, weight=1)

        # Info del modelo actual
        model_info_frame = ttk.LabelFrame(form_frame, text="Información del Modelo")
        model_info_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(model_info_frame, text="Nombre del archivo:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_filename = ttk.Label(model_info_frame, text="")
        self.lbl_filename.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(model_info_frame, text="Vértices:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_vertices = ttk.Label(model_info_frame, text="")
        self.lbl_vertices.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(model_info_frame, text="Caras:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_faces = ttk.Label(model_info_frame, text="")
        self.lbl_faces.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(model_info_frame, text="Es imprimible:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_watertight = ttk.Label(model_info_frame, text="")
        self.lbl_watertight.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # Formulario de etiquetado
        label_frame = ttk.LabelFrame(form_frame, text="Etiquetado")
        label_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tipo de florero
        ttk.Label(label_frame, text="Tipo de florero:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.cmb_type = ttk.Combobox(label_frame, values=self.vase_types)
        self.cmb_type.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Estilo
        ttk.Label(label_frame, text="Estilo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.cmb_style = ttk.Combobox(label_frame, values=self.vase_styles)
        self.cmb_style.grid(row=1, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Complejidad
        ttk.Label(label_frame, text="Complejidad:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.cmb_complexity = ttk.Combobox(label_frame, values=self.complexity_levels)
        self.cmb_complexity.grid(row=2, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Grosor de pared
        ttk.Label(label_frame, text="Grosor de pared (mm):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_wall_thickness = ttk.Entry(label_frame)
        self.txt_wall_thickness.grid(row=3, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Necesita soportes
        ttk.Label(label_frame, text="¿Necesita soportes?:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.var_needs_supports = tk.StringVar()
        self.rad_yes = ttk.Radiobutton(label_frame, text="Sí", variable=self.var_needs_supports, value="Sí")
        self.rad_yes.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        self.rad_no = ttk.Radiobutton(label_frame, text="No", variable=self.var_needs_supports, value="No")
        self.rad_no.grid(row=4, column=1, sticky=tk.E, padx=5, pady=5)

        # Tiempo estimado de impresión
        ttk.Label(label_frame, text="Tiempo de impresión (h):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_print_time = ttk.Entry(label_frame)
        self.txt_print_time.grid(row=5, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Uso de filamento
        ttk.Label(label_frame, text="Filamento (g):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_filament = ttk.Entry(label_frame)
        self.txt_filament.grid(row=6, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Etiquetas
        ttk.Label(label_frame, text="Etiquetas:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_tags = ttk.Entry(label_frame)
        self.txt_tags.grid(row=7, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Notas
        ttk.Label(label_frame, text="Notas:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_notes = tk.Text(label_frame, height=4, width=20)
        self.txt_notes.grid(row=8, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Botones de navegación
        nav_frame = ttk.Frame(form_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)

        self.btn_prev = ttk.Button(nav_frame, text="Anterior", command=self.prev_model)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(nav_frame, text="Guardar", command=self.save_labels)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_next = ttk.Button(nav_frame, text="Siguiente", command=self.next_model)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        # Contador de progreso
        self.lbl_progress = ttk.Label(form_frame, text="")
        self.lbl_progress.pack(pady=5)

    def load_model(self, index):
        """Carga un modelo y lo visualiza"""
        if not self.model_files or index < 0 or index >= len(self.model_files):
            return

        self.current_model_index = index
        file_path = self.model_files[index]
        filename = os.path.basename(file_path)

        try:
            # Cargar modelo con trimesh
            self.current_model = trimesh.load(file_path)

            # Actualizar información del modelo
            self.lbl_filename.config(text=filename)
            self.lbl_vertices.config(text=str(len(self.current_model.vertices)))
            self.lbl_faces.config(text=str(len(self.current_model.faces)))
            self.lbl_watertight.config(text="Sí" if self.current_model.is_watertight else "No")

            # Visualizar modelo
            self.visualize_model()

            # Cargar etiquetas existentes
            self.load_existing_labels(filename)

            # Actualizar contador de progreso
            self.lbl_progress.config(text=f"Modelo {index + 1} de {len(self.model_files)}")

        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el modelo: {str(e)}")

    def visualize_model(self):
        """Visualiza el modelo actual en múltiples vistas"""
        if self.current_model is None:
            return

        # Limpiar figura actual
        self.fig.clear()

        # Crear escena
        scene = trimesh.Scene(self.current_model)

        # Vista frontal
        ax1 = self.fig.add_subplot(221)
        scene.camera_transform = trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0], scene.centroid)
        img_front = scene.save_image(resolution=[200, 200])
        from PIL import Image
        import io
        ax1.imshow(np.array(Image.open(io.BytesIO(img_front))))
        ax1.set_title("Frontal")
        ax1.axis('off')

        # Vista lateral
        ax2 = self.fig.add_subplot(222)
        scene.camera_transform = trimesh.transformations.rotation_matrix(
            0, [0, 1, 0], scene.centroid)
        img_side = scene.save_image(resolution=[200, 200])
        ax2.imshow(np.array(Image.open(io.BytesIO(img_side))))
        ax2.set_title("Lateral")
        ax2.axis('off')

        # Vista superior
        ax3 = self.fig.add_subplot(223)
        scene.camera_transform = trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0], scene.centroid)
        img_top = scene.save_image(resolution=[200, 200])
        ax3.imshow(np.array(Image.open(io.BytesIO(img_top))))
        ax3.set_title("Superior")
        ax3.axis('off')

        # Vista isométrica
        ax4 = self.fig.add_subplot(224)
        scene.camera_transform = trimesh.transformations.rotation_matrix(
            np.pi / 4, [1, 0, 0], scene.centroid)
        scene.camera_transform = trimesh.transformations.rotation_matrix(
            np.pi / 4, [0, 0, 1], scene.centroid).dot(scene.camera_transform)
        img_iso = scene.save_image(resolution=[200, 200])
        ax4.imshow(np.array(Image.open(io.BytesIO(img_iso))))
        ax4.set_title("Isométrica")
        ax4.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

    def load_existing_labels(self, filename):
        """Carga etiquetas existentes para el modelo actual"""
        # Limpiar formulario
        self.cmb_type.set("")
        self.cmb_style.set("")
        self.cmb_complexity.set("")
        self.txt_wall_thickness.delete(0, tk.END)
        self.var_needs_supports.set("")
        self.txt_print_time.delete(0, tk.END)
        self.txt_filament.delete(0, tk.END)
        self.txt_tags.delete(0, tk.END)
        self.txt_notes.delete("1.0", tk.END)

        # Cargar datos existentes si los hay
        if filename in self.metadata:
            data = self.metadata[filename]
            self.cmb_type.set(data.get('type', ''))
            self.cmb_style.set(data.get('style', ''))
            self.cmb_complexity.set(data.get('complexity', ''))
            self.txt_wall_thickness.insert(0, data.get('wall_thickness', ''))
            self.var_needs_supports.set(data.get('needs_supports', ''))
            self.txt_print_time.insert(0, data.get('print_time_estimate', ''))
            self.txt_filament.insert(0, data.get('filament_usage', ''))
            self.txt_tags.insert(0, data.get('tags', ''))
            self.txt_notes.insert("1.0", data.get('notes', ''))

    def save_labels(self):
        """Guarda las etiquetas del modelo actual"""
        if not self.model_files or self.current_model_index >= len(self.model_files):
            return

        filename = os.path.basename(self.model_files[self.current_model_index])

        # Recopilar datos
        data = {
            'id': str(self.current_model_index + 1),
            'filename': filename,
            'type': self.cmb_type.get(),
            'style': self.cmb_style.get(),
            'complexity': self.cmb_complexity.get(),
            'wall_thickness': self.txt_wall_thickness.get(),
            'needs_supports': self.var_needs_supports.get(),
            'print_time_estimate': self.txt_print_time.get(),
            'filament_usage': self.txt_filament.get(),
            'tags': self.txt_tags.get(),
            'notes': self.txt_notes.get("1.0", tk.END).strip()
        }

        # Guardar en diccionario
        self.metadata[filename] = data

        # Guardar en archivo CSV
        fieldnames = ['id', 'filename', 'type', 'style', 'complexity',
                      'wall_thickness', 'needs_supports', 'print_time_estimate',
                      'filament_usage', 'tags', 'notes']

        rows = []
        for key in self.metadata:
            rows.append(self.metadata[key])

        with open(self.metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        messagebox.showinfo("Guardado", f"Etiquetas guardadas para {filename}")

    def next_model(self):
        """Carga el siguiente modelo"""
        self.save_labels()
        if self.current_model_index + 1 < len(self.model_files):
            self.load_model(self.current_model_index + 1)

    def prev_model(self):
        """Carga el modelo anterior"""
        self.save_labels()
        if self.current_model_index > 0:
            self.load_model(self.current_model_index - 1)


def main():
    if len(sys.argv) > 2:
        models_dir = sys.argv[1]
        metadata_file = sys.argv[2]
    else:
        models_dir = filedialog.askdirectory(title="Selecciona la carpeta de modelos 3D")
        if not models_dir:
            return

        metadata_file = filedialog.asksaveasfilename(
            title="Selecciona el archivo de metadatos",
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv")]
        )
        if not metadata_file:
            return

    root = tk.Tk()
    app = VaseLabelingTool(root, models_dir, metadata_file)
    root.mainloop()


if __name__ == "__main__":
    main()