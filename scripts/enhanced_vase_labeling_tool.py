# enhanced_vase_labeling_tool.py corregido
import os
import csv
import sys
import json
import numpy as np
import pandas as pd
import trimesh
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from concurrent.futures import ThreadPoolExecutor
import threading
import time  # Importar time para time.time()


class EnhancedVaseLabelingTool:
    def __init__(self, root, models_dir, metadata_file):
        self.root = root
        self.root.title("Herramienta Avanzada de Etiquetado de Floreros 3D")
        self.root.geometry("1400x900")

        # Configurar estilo
        style = ttk.Style()
        style.theme_use('clam')  # Tema moderno

        self.models_dir = models_dir
        self.metadata_file = metadata_file

        # Para procesamiento en segundo plano
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.load_futures = []

        # Estado de filtrado y búsqueda
        self.filter_criteria = {}
        self.search_term = ""

        # Listas para etiquetado
        self.vase_types = [
            "Estándar", "Con cuello", "Cilíndrico", "Esférico", "Asimétrico",
            "Artístico", "Minimalista", "Geométrico", "Con mango", "Con tapa"
        ]
        self.vase_styles = [
            "Liso", "Geométrico", "Orgánico", "Floral", "Texturizado", "Módular",
            "Moderno", "Clásico", "Art Deco", "Abstracto", "Minimalista", "Industrial"
        ]
        self.complexity_levels = [
            "1 - Simple", "2 - Media", "3 - Compleja", "4 - Muy compleja"
        ]
        self.printability_levels = [
            "1 - Fácil de imprimir", "2 - Impresión estándar",
            "3 - Impresión desafiante", "4 - Difícil de imprimir"
        ]

        # Cargar o crear archivo de metadatos
        self.load_metadata()

        # Variables de etiquetado
        self.current_model_index = 0
        self.current_model = None
        self.model_files = self.get_model_files()
        self.filtered_files = self.model_files.copy()

        # Crear interfaz
        self.create_widgets()

        # Cargar primer modelo
        if self.filtered_files:
            self.load_model(0)

    def get_model_files(self):
        """Obtiene lista de archivos de modelos 3D en el directorio"""
        model_files = []
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.lower().endswith(('.stl', '.obj', '.ply')):
                    model_files.append(os.path.join(root, file))

        # Ordenar para consistencia
        model_files.sort()
        return model_files

    def load_metadata(self):
        """Carga o crea archivo de metadatos"""
        self.metadata = {}

        if os.path.exists(self.metadata_file):
            try:
                # Intentar cargar como CSV
                if self.metadata_file.lower().endswith('.csv'):
                    with open(self.metadata_file, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            self.metadata[row['filename']] = row
                # Intentar cargar como JSON
                elif self.metadata_file.lower().endswith('.json'):
                    with open(self.metadata_file, 'r', encoding='utf-8') as jsonfile:
                        self.metadata = json.load(jsonfile)
            except Exception as e:
                messagebox.showerror("Error de carga", f"Error al cargar metadatos: {str(e)}")
                self.metadata = {}

        # Crear archivo con encabezados si no existe
        if not os.path.exists(self.metadata_file):
            if self.metadata_file.lower().endswith('.csv'):
                with open(self.metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['id', 'filename', 'type', 'style', 'complexity',
                                  'printability', 'wall_thickness', 'needs_supports',
                                  'print_time_estimate', 'filament_usage', 'tags', 'notes',
                                  'is_watertight', 'volume', 'faces', 'vertices']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
            elif self.metadata_file.lower().endswith('.json'):
                with open(self.metadata_file, 'w', encoding='utf-8') as jsonfile:
                    json.dump({}, jsonfile)

    def create_widgets(self):
        """Crea la interfaz de usuario mejorada"""
        # Marco principal
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel izquierdo (navegación y filtrado)
        left_frame = ttk.Frame(main_paned, width=250)
        main_paned.add(left_frame, weight=1)

        # Panel de filtros y búsqueda
        filter_frame = ttk.LabelFrame(left_frame, text="Búsqueda y Filtros")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(filter_frame, text="Buscar:").pack(anchor=tk.W, padx=5, pady=2)
        self.search_entry = ttk.Entry(filter_frame)
        self.search_entry.pack(fill=tk.X, padx=5, pady=2)
        self.search_entry.bind("<Return>", lambda e: self.apply_filters())

        ttk.Label(filter_frame, text="Tipo:").pack(anchor=tk.W, padx=5, pady=2)
        self.filter_type = ttk.Combobox(filter_frame, values=["Todos"] + self.vase_types)
        self.filter_type.current(0)
        self.filter_type.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(filter_frame, text="Complejidad:").pack(anchor=tk.W, padx=5, pady=2)
        self.filter_complexity = ttk.Combobox(filter_frame, values=["Todos"] + self.complexity_levels)
        self.filter_complexity.current(0)
        self.filter_complexity.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(filter_frame, text="Estado:").pack(anchor=tk.W, padx=5, pady=2)
        self.filter_status = ttk.Combobox(filter_frame, values=["Todos", "Etiquetados", "Sin etiquetar"])
        self.filter_status.current(0)
        self.filter_status.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(filter_frame, text="Aplicar Filtros", command=self.apply_filters).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(filter_frame, text="Limpiar Filtros", command=self.clear_filters).pack(fill=tk.X, padx=5, pady=5)

        # Lista de archivos
        file_frame = ttk.LabelFrame(left_frame, text="Modelos")
        file_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.file_listbox = tk.Listbox(file_frame, selectmode=tk.SINGLE)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        # Contador e info
        self.status_var = tk.StringVar()
        status_label = ttk.Label(left_frame, textvariable=self.status_var)
        status_label.pack(fill=tk.X, padx=5, pady=5)

        # Panel central (visualización)
        center_frame = ttk.Frame(main_paned)
        main_paned.add(center_frame, weight=3)

        # Configurar visualización del modelo
        vis_frame = ttk.LabelFrame(center_frame, text="Visualización 3D")
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = plt.figure(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Botones de visualización
        vis_buttons = ttk.Frame(vis_frame)
        vis_buttons.pack(fill=tk.X)

        ttk.Button(vis_buttons, text="Vista Frontal", command=lambda: self.change_view("front")).pack(side=tk.LEFT,
                                                                                                      padx=2)
        ttk.Button(vis_buttons, text="Vista Superior", command=lambda: self.change_view("top")).pack(side=tk.LEFT,
                                                                                                     padx=2)
        ttk.Button(vis_buttons, text="Vista Lateral", command=lambda: self.change_view("side")).pack(side=tk.LEFT,
                                                                                                     padx=2)
        ttk.Button(vis_buttons, text="Isométrica", command=lambda: self.change_view("iso")).pack(side=tk.LEFT, padx=2)
        ttk.Button(vis_buttons, text="Rotar", command=self.rotate_model).pack(side=tk.LEFT, padx=2)

        # Panel derecho (formulario)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # Info del modelo actual
        model_info_frame = ttk.LabelFrame(right_frame, text="Información del Modelo")
        model_info_frame.pack(fill=tk.X, padx=5, pady=5)

        # Crear grid de info con 2 columnas
        info_grid = ttk.Frame(model_info_frame)
        info_grid.pack(fill=tk.X, pady=5)

        ttk.Label(info_grid, text="Nombre:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_filename = ttk.Label(info_grid, text="", wraplength=200)
        self.lbl_filename.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Vértices:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_vertices = ttk.Label(info_grid, text="")
        self.lbl_vertices.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Caras:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_faces = ttk.Label(info_grid, text="")
        self.lbl_faces.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Es imprimible:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_watertight = ttk.Label(info_grid, text="")
        self.lbl_watertight.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Volumen:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_volume = ttk.Label(info_grid, text="")
        self.lbl_volume.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Dimensiones:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_dimensions = ttk.Label(info_grid, text="")
        self.lbl_dimensions.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        # Formulario de etiquetado
        label_frame = ttk.LabelFrame(right_frame, text="Etiquetado")
        label_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Usar grid para mejor organización
        ttk.Label(label_frame, text="Tipo de florero:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.cmb_type = ttk.Combobox(label_frame, values=self.vase_types)
        self.cmb_type.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(label_frame, text="Estilo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.cmb_style = ttk.Combobox(label_frame, values=self.vase_styles)
        self.cmb_style.grid(row=1, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(label_frame, text="Complejidad:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.cmb_complexity = ttk.Combobox(label_frame, values=self.complexity_levels)
        self.cmb_complexity.grid(row=2, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(label_frame, text="Imprimibilidad:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.cmb_printability = ttk.Combobox(label_frame, values=self.printability_levels)
        self.cmb_printability.grid(row=3, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(label_frame, text="Grosor de pared (mm):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_wall_thickness = ttk.Entry(label_frame)
        self.txt_wall_thickness.grid(row=4, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(label_frame, text="¿Necesita soportes?:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.var_needs_supports = tk.StringVar()
        support_frame = ttk.Frame(label_frame)
        support_frame.grid(row=5, column=1, sticky=tk.W)
        self.rad_yes = ttk.Radiobutton(support_frame, text="Sí", variable=self.var_needs_supports, value="Sí")
        self.rad_yes.pack(side=tk.LEFT)
        self.rad_no = ttk.Radiobutton(support_frame, text="No", variable=self.var_needs_supports, value="No")
        self.rad_no.pack(side=tk.LEFT)

        ttk.Label(label_frame, text="Tiempo de impresión (h):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_print_time = ttk.Entry(label_frame)
        self.txt_print_time.grid(row=6, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(label_frame, text="Filamento (g):").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_filament = ttk.Entry(label_frame)
        self.txt_filament.grid(row=7, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(label_frame, text="Etiquetas:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_tags = ttk.Entry(label_frame)
        self.txt_tags.grid(row=8, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(label_frame, text="Notas:").grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_notes = tk.Text(label_frame, height=4, width=20)
        self.txt_notes.grid(row=9, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Botones de navegación y acción
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)

        self.btn_prev = ttk.Button(nav_frame, text="Anterior", command=self.prev_model)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.btn_save = ttk.Button(nav_frame, text="Guardar", command=self.save_labels)
        self.btn_save.pack(side=tk.LEFT, padx=2)

        self.btn_next = ttk.Button(nav_frame, text="Siguiente", command=self.next_model)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        self.btn_skip = ttk.Button(nav_frame, text="Omitir", command=lambda: self.next_model(skip=True))
        self.btn_skip.pack(side=tk.LEFT, padx=2)

        # Botones adicionales
        extra_frame = ttk.Frame(right_frame)
        extra_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(extra_frame, text="Análisis Automático", command=self.auto_analyze).pack(side=tk.LEFT, padx=2)
        ttk.Button(extra_frame, text="Guardar Todo", command=self.save_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(extra_frame, text="Estadísticas", command=self.show_stats).pack(side=tk.LEFT, padx=2)

        # Actualizar lista de archivos
        self.update_file_list()

    def update_file_list(self):
        """Actualiza la lista de archivos según filtros"""
        self.file_listbox.delete(0, tk.END)

        for idx, file_path in enumerate(self.filtered_files):
            filename = os.path.basename(file_path)
            # Añadir indicador si está etiquetado
            if filename in self.metadata:
                item_text = f"✓ {filename}"
            else:
                item_text = f"□ {filename}"
            self.file_listbox.insert(tk.END, item_text)

        # Actualizar contador
        self.update_status()

    def update_status(self):
        """Actualiza el texto de estado"""
        total = len(self.model_files)
        filtered = len(self.filtered_files)
        labeled = sum(1 for f in [os.path.basename(f) for f in self.model_files] if f in self.metadata)

        if total > 0:
            self.status_var.set(
                f"Total: {total} | Filtrados: {filtered} | Etiquetados: {labeled} ({labeled / total * 100:.1f}%)")
        else:
            self.status_var.set("No se encontraron modelos")

    def apply_filters(self):
        """Aplica filtros a la lista de archivos"""
        self.filtered_files = []
        search_term = self.search_entry.get().lower()
        filter_type = self.filter_type.get()
        filter_complexity = self.filter_complexity.get()
        filter_status = self.filter_status.get()

        for file_path in self.model_files:
            filename = os.path.basename(file_path)
            include = True

            # Filtrar por término de búsqueda
            if search_term and search_term not in filename.lower():
                include = False

            # Filtrar por tipo
            if filter_type != "Todos" and filename in self.metadata:
                if self.metadata[filename].get('type', '') != filter_type:
                    include = False

            # Filtrar por complejidad
            if filter_complexity != "Todos" and filename in self.metadata:
                if self.metadata[filename].get('complexity', '') != filter_complexity:
                    include = False

            # Filtrar por estado de etiquetado
            if filter_status == "Etiquetados" and filename not in self.metadata:
                include = False
            elif filter_status == "Sin etiquetar" and filename in self.metadata:
                include = False

            if include:
                self.filtered_files.append(file_path)

        self.update_file_list()

        # Si hay resultados, seleccionar el primero
        if self.filtered_files:
            self.load_model(0)

    def clear_filters(self):
        """Limpia todos los filtros"""
        self.search_entry.delete(0, tk.END)
        self.filter_type.current(0)
        self.filter_complexity.current(0)
        self.filter_status.current(0)

        self.filtered_files = self.model_files.copy()
        self.update_file_list()

    def on_file_select(self, event):
        """Maneja la selección de un archivo en la lista"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_model(index)

    def load_model(self, index):
        """Carga un modelo y lo visualiza"""
        if not self.filtered_files or index < 0 or index >= len(self.filtered_files):
            return

        # Guardar etiquetas del modelo actual antes de cambiar
        if hasattr(self, 'current_model_index') and self.current_model_index < len(self.filtered_files):
            self.save_labels(silent=True)

        self.current_model_index = index
        file_path = self.filtered_files[index]
        filename = os.path.basename(file_path)

        # Seleccionar en la lista
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(index)
        self.file_listbox.see(index)

        # Mostrar indicador de carga
        self.root.config(cursor="wait")
        self.lbl_filename.config(text=f"Cargando {filename}...")
        self.root.update()

        try:
            # Cargar modelo con trimesh (en segundo plano)
            def load_in_background():
                try:
                    model = trimesh.load(file_path)

                    # Calcular propiedades adicionales
                    watertight = model.is_watertight
                    volume = model.volume if watertight else None
                    dimensions = model.bounding_box.extents

                    return {
                        'model': model,
                        'vertices': len(model.vertices),
                        'faces': len(model.faces),
                        'watertight': watertight,
                        'volume': volume,
                        'dimensions': dimensions
                    }
                except Exception as e:
                    return {'error': str(e)}

            # Iniciar carga en segundo plano
            future = self.executor.submit(load_in_background)
            self.load_futures.append(future)

            def on_model_loaded(future):
                result = future.result()

                if 'error' in result:
                    messagebox.showerror("Error", f"Error al cargar el modelo: {result['error']}")
                    return

                self.current_model = result['model']

                # Actualizar información del modelo
                self.lbl_filename.config(text=filename)
                self.lbl_vertices.config(text=f"{result['vertices']:,}")
                self.lbl_faces.config(text=f"{result['faces']:,}")
                self.lbl_watertight.config(text="Sí" if result['watertight'] else "No")

                vol_text = f"{result['volume']:.2f} cm³" if result['volume'] is not None else "N/A"
                self.lbl_volume.config(text=vol_text)

                dims = result['dimensions']
                dim_text = f"{dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f} cm"
                self.lbl_dimensions.config(text=dim_text)

                # Visualizar modelo
                self.visualize_model()

                # Cargar etiquetas existentes
                self.load_existing_labels(filename)

                # Restaurar cursor
                self.root.config(cursor="")

            # Configurar callback cuando termine la carga
            future.add_done_callback(lambda f: self.root.after(0, on_model_loaded, f))

        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el modelo: {str(e)}")
            self.root.config(cursor="")

    def visualize_model(self, view="iso"):
        """Visualiza el modelo actual usando matplotlib"""
        if self.current_model is None:
            return

        # Limpiar figura anterior
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')

        try:
            # Mostrar la malla 3D
            vertices = self.current_model.vertices
            faces = self.current_model.faces

            # Crear polígonos
            poly3d = [[vertices[vertex] for vertex in face] for face in faces]

            # Usar Poly3DCollection para mejor rendimiento con modelos grandes
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh = Poly3DCollection(poly3d, alpha=0.5, linewidths=0.5, edgecolor='#555555')
            mesh.set_facecolor('#1e88e5')  # Color azul
            ax.add_collection3d(mesh)

            # Autoescalar la vista
            scale = self.current_model.vertices.flatten()
            ax.auto_scale_xyz(scale, scale, scale)

            # Configurar vista
            if view == "front":
                ax.view_init(elev=0, azim=0)
            elif view == "top":
                ax.view_init(elev=90, azim=0)
            elif view == "side":
                ax.view_init(elev=0, azim=90)
            else:  # vista isométrica por defecto
                ax.view_init(elev=30, azim=45)

            ax.set_title(os.path.basename(self.filtered_files[self.current_model_index]))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error de visualización", f"No se pudo visualizar el modelo: {str(e)}")
            # Mostrar mensaje de error en el lienzo
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error al visualizar:\n{str(e)}",
                    ha='center', va='center')
            self.canvas.draw()
    def change_view(self, view):
        """Cambia la vista del modelo"""
        self.visualize_model(view)

    def rotate_model(self):
        """Rota el modelo"""
        self.visualize_model("rotate")

    def load_existing_labels(self, filename):
        """Carga etiquetas existentes para el modelo actual"""
        # Limpiar formulario
        self.cmb_type.set("")
        self.cmb_style.set("")
        self.cmb_complexity.set("")
        self.cmb_printability.set("")
        self.txt_wall_thickness.delete(0, tk.END)
        self.var_needs_supports.set("")
        self.txt_print_time.delete(0, tk.END)
        self.txt_filament.delete(0, tk.END)
        self.txt_tags.delete(0, tk.END)
        self.txt_notes.delete("1.0", tk.END)

        # Cargar datos existentes si los hay
        # Cargar datos existentes si los hay
        if filename in self.metadata:
            data = self.metadata[filename]
            self.cmb_type.set(data.get('type', ''))
            self.cmb_style.set(data.get('style', ''))
            self.cmb_complexity.set(data.get('complexity', ''))
            self.cmb_printability.set(data.get('printability', ''))
            self.txt_wall_thickness.insert(0, data.get('wall_thickness', ''))
            self.var_needs_supports.set(data.get('needs_supports', ''))
            self.txt_print_time.insert(0, data.get('print_time_estimate', ''))
            self.txt_filament.insert(0, data.get('filament_usage', ''))
            self.txt_tags.insert(0, data.get('tags', ''))
            self.txt_notes.insert("1.0", data.get('notes', ''))

    def save_labels(self, silent=False):
        """Guarda las etiquetas del modelo actual"""
        if not self.filtered_files or self.current_model_index >= len(self.filtered_files):
            return

        file_path = self.filtered_files[self.current_model_index]
        filename = os.path.basename(file_path)

        # Recopilar datos
        data = {
            'id': str(self.current_model_index + 1),
            'filename': filename,
            'type': self.cmb_type.get(),
            'style': self.cmb_style.get(),
            'complexity': self.cmb_complexity.get(),
            'printability': self.cmb_printability.get(),
            'wall_thickness': self.txt_wall_thickness.get(),
            'needs_supports': self.var_needs_supports.get(),
            'print_time_estimate': self.txt_print_time.get(),
            'filament_usage': self.txt_filament.get(),
            'tags': self.txt_tags.get(),
            'notes': self.txt_notes.get("1.0", tk.END).strip()
        }

        # Añadir propiedades medidas
        if hasattr(self, 'current_model') and self.current_model is not None:
            data.update({
                'vertices': str(len(self.current_model.vertices)),
                'faces': str(len(self.current_model.faces)),
                'is_watertight': str(self.current_model.is_watertight),
                'volume': str(self.current_model.volume if self.current_model.is_watertight else ""),
            })

        # Guardar en diccionario
        self.metadata[filename] = data

        # Guardar en archivo
        self.save_metadata_file()

        if not silent:
            messagebox.showinfo("Guardado", f"Etiquetas guardadas para {filename}")

        # Actualizar estado
        self.update_file_list()

    def save_metadata_file(self):
        """Guarda el archivo de metadatos"""
        try:
            if self.metadata_file.lower().endswith('.csv'):
                # Obtener todos los campos posibles
                fieldnames = set()
                for data in self.metadata.values():
                    fieldnames.update(data.keys())

                fieldnames = sorted(list(fieldnames))

                with open(self.metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for filename in sorted(self.metadata.keys()):
                        writer.writerow(self.metadata[filename])

            elif self.metadata_file.lower().endswith('.json'):
                with open(self.metadata_file, 'w', encoding='utf-8') as jsonfile:
                    json.dump(self.metadata, jsonfile, indent=2)

        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar metadatos: {str(e)}")

    def next_model(self, skip=False):
        """Carga el siguiente modelo"""
        if not skip:
            self.save_labels(silent=True)

        if self.current_model_index + 1 < len(self.filtered_files):
            self.load_model(self.current_model_index + 1)
        else:
            messagebox.showinfo("Fin", "Has llegado al último modelo")

    def prev_model(self):
        """Carga el modelo anterior"""
        self.save_labels(silent=True)

        if self.current_model_index > 0:
            self.load_model(self.current_model_index - 1)
        else:
            messagebox.showinfo("Inicio", "Estás en el primer modelo")

    def auto_analyze(self):
        """Realiza un análisis automático del modelo actual"""
        if self.current_model is None:
            messagebox.showwarning("Aviso", "No hay modelo cargado")
            return

        # Analizar propiedades del modelo
        properties = {}

        # Complejidad basada en número de caras
        faces = len(self.current_model.faces)
        if faces < 5000:
            properties['complexity'] = self.complexity_levels[0]  # Simple
        elif faces < 20000:
            properties['complexity'] = self.complexity_levels[1]  # Media
        elif faces < 50000:
            properties['complexity'] = self.complexity_levels[2]  # Compleja
        else:
            properties['complexity'] = self.complexity_levels[3]  # Muy compleja

        # Imprimibilidad basada en estanqueidad y proporción
        is_watertight = self.current_model.is_watertight
        dims = self.current_model.bounding_box.extents
        height = max(dims)
        width = min(dims)
        aspect_ratio = height / width if width > 0 else float('inf')

        if is_watertight and aspect_ratio < 3:
            properties['printability'] = self.printability_levels[0]  # Fácil
        elif is_watertight and aspect_ratio < 5:
            properties['printability'] = self.printability_levels[1]  # Estándar
        elif is_watertight:
            properties['printability'] = self.printability_levels[2]  # Desafiante
        else:
            properties['printability'] = self.printability_levels[3]  # Difícil

        # Soporte basado en ángulos y voladizos
        needs_supports = "No"
        if aspect_ratio > 4 or not is_watertight:
            needs_supports = "Sí"

        # Tiempo de impresión estimado (muy aproximado)
        volume = self.current_model.volume if is_watertight else 0
        if volume > 0:
            # Fórmula muy básica: 1 cm³ ≈ 10 minutos (depende de muchos factores)
            print_time = volume / 6  # Horas
            properties['print_time_estimate'] = f"{print_time:.1f}"

            # Filamento: 1 cm³ ≈ 1.25g (PLA)
            filament = volume * 1.25
            properties['filament_usage'] = f"{filament:.0f}"

        # Aplicar propiedades
        self.cmb_complexity.set(properties.get('complexity', ''))
        self.cmb_printability.set(properties.get('printability', ''))
        self.var_needs_supports.set(needs_supports)

        if 'print_time_estimate' in properties:
            self.txt_print_time.delete(0, tk.END)
            self.txt_print_time.insert(0, properties['print_time_estimate'])

        if 'filament_usage' in properties:
            self.txt_filament.delete(0, tk.END)
            self.txt_filament.insert(0, properties['filament_usage'])

        messagebox.showinfo("Análisis Completado",
                            "Se ha realizado un análisis automático del modelo.\n"
                            "Los resultados son aproximados y deben revisarse.")

    def save_all(self):
        """Guarda todos los metadatos"""
        self.save_labels(silent=True)
        self.save_metadata_file()
        messagebox.showinfo("Guardado", "Todos los datos han sido guardados")

    def show_stats(self):
        """Muestra estadísticas del conjunto de datos"""
        # Contar cantidad por tipo
        type_counts = {}
        for data in self.metadata.values():
            type_name = data.get('type', 'Sin etiquetar')
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Contar por complejidad
        complexity_counts = {}
        for data in self.metadata.values():
            complexity = data.get('complexity', 'Sin etiquetar')
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # Contar necesidad de soportes
        support_counts = {'Sí': 0, 'No': 0, 'Sin etiquetar': 0}
        for data in self.metadata.values():
            support = data.get('needs_supports', 'Sin etiquetar')
            support_counts[support] = support_counts.get(support, 0) + 1

        # Mostrar estadísticas en una ventana
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Estadísticas del Conjunto de Datos")
        stats_window.geometry("600x500")

        ttk.Label(stats_window, text="Estadísticas de Etiquetado",
                  font=("Helvetica", 14, "bold")).pack(pady=10)

        # Total
        total_labeled = len(self.metadata)
        total_models = len(self.model_files)
        progress = total_labeled / total_models * 100 if total_models > 0 else 0

        ttk.Label(stats_window,
                  text=f"Progreso: {total_labeled} de {total_models} modelos etiquetados ({progress:.1f}%)"
                  ).pack(pady=5)

        # Distribución por tipo
        ttk.Label(stats_window, text="Distribución por Tipo:",
                  font=("Helvetica", 12, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percent = count / total_labeled * 100 if total_labeled > 0 else 0
            ttk.Label(stats_window,
                      text=f"- {type_name}: {count} modelos ({percent:.1f}%)"
                      ).pack(anchor=tk.W, padx=20)

        # Distribución por complejidad
        ttk.Label(stats_window, text="Distribución por Complejidad:",
                  font=("Helvetica", 12, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        for complexity, count in sorted(complexity_counts.items(), key=lambda x: x[1], reverse=True):
            percent = count / total_labeled * 100 if total_labeled > 0 else 0
            ttk.Label(stats_window,
                      text=f"- {complexity}: {count} modelos ({percent:.1f}%)"
                      ).pack(anchor=tk.W, padx=20)

        # Distribución por soporte
        ttk.Label(stats_window, text="Necesidad de Soportes:",
                  font=("Helvetica", 12, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        for support, count in support_counts.items():
            if support == 'Sin etiquetar' and count == 0:
                continue
            percent = count / total_labeled * 100 if total_labeled > 0 else 0
            ttk.Label(stats_window,
                      text=f"- {support}: {count} modelos ({percent:.1f}%)"
                      ).pack(anchor=tk.W, padx=20)

        # Botón para cerrar
        ttk.Button(stats_window, text="Cerrar", command=stats_window.destroy).pack(pady=10)

def main():
    # Obtener argumentos
    if len(sys.argv) > 2:
        models_dir = sys.argv[1]
        metadata_file = sys.argv[2]
    else:
        # Mostrar diálogo para seleccionar directorio y archivo
        root = tk.Tk()
        root.withdraw()  # Ocultar ventana principal

        # Seleccionar directorio
        models_dir = filedialog.askdirectory(title="Seleccione el directorio con modelos 3D")
        if not models_dir:
            return

        # Seleccionar archivo de metadatos
        metadata_file = filedialog.asksaveasfilename(
            title="Seleccione o cree archivo de metadatos",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")]
        )
        if not metadata_file:
            return

    # Crear y ejecutar aplicación
    root = tk.Tk()
    app = EnhancedVaseLabelingTool(root, models_dir, metadata_file)
    root.mainloop()

if __name__ == "__main__":
    main()