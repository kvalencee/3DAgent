# guarda este archivo como setup_training_data.py en el directorio raíz de tu proyecto
import os
import numpy as np
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
import random

# Rutas
voxels_dir = "C:\\Users\\keval\\OneDrive\\Documentos\\decorai\\data\\processed\\voxels"
output_dir = "C:\\Users\\keval\\OneDrive\\Documentos\\decorai\\data\\training_ready"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Crear directorios
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Buscar todos los archivos .npy recursivamente
all_npy_files = []
for root, dirs, files in os.walk(voxels_dir):
    for file in files:
        if file.lower().endswith('.npy'):
            all_npy_files.append(os.path.join(root, file))

print(f"Encontrados {len(all_npy_files)} archivos .npy")

if len(all_npy_files) == 0:
    print("No se encontraron archivos para procesar.")
    exit()

# Dividir en train/test (80/20)
train_files, test_files = train_test_split(all_npy_files, test_size=0.2, random_state=42)

print(f"Archivos de entrenamiento: {len(train_files)}")
print(f"Archivos de prueba: {len(test_files)}")

# Copiar archivos para entrenamiento
for src_path in train_files:
    # Obtener la categoría (suponiendo que está en el penúltimo nivel de directorio)
    parts = src_path.split(os.sep)
    if len(parts) >= 2:
        category = parts[-3]  # Ajusta esto según la estructura real
    else:
        category = "misc"

    # Crear directorio de categoría si no existe
    category_dir = os.path.join(train_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    # Nombre de archivo destino
    dst_path = os.path.join(category_dir, os.path.basename(src_path))

    # Copiar archivo
    shutil.copy2(src_path, dst_path)
    print(f"Copiado: {src_path} -> {dst_path}")

# Copiar archivos para prueba
for src_path in test_files:
    # Obtener la categoría (suponiendo que está en el penúltimo nivel de directorio)
    parts = src_path.split(os.sep)
    if len(parts) >= 2:
        category = parts[-3]  # Ajusta esto según la estructura real
    else:
        category = "misc"

    # Crear directorio de categoría si no existe
    category_dir = os.path.join(test_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    # Nombre de archivo destino
    dst_path = os.path.join(category_dir, os.path.basename(src_path))

    # Copiar archivo
    shutil.copy2(src_path, dst_path)
    print(f"Copiado: {src_path} -> {dst_path}")

print("\nProceso completado.")
print(f"Archivos de entrenamiento: {len(train_files)}")
print(f"Archivos de prueba: {len(test_files)}")