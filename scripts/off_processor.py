# off_processor.py
import os
import argparse
import numpy as np
import trimesh
from tqdm import tqdm


def convert_off_to_stl(off_file, output_dir=None):
    """
    Convierte un archivo OFF a STL

    Args:
        off_file: Ruta al archivo OFF
        output_dir: Directorio de salida (opcional)

    Returns:
        str: Ruta al archivo STL generado
    """
    if output_dir is None:
        output_dir = os.path.dirname(off_file)

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Cargar malla OFF
    mesh = trimesh.load(off_file)

    # Nombre de archivo de salida
    base_name = os.path.basename(off_file)
    stl_name = os.path.splitext(base_name)[0] + ".stl"
    stl_path = os.path.join(output_dir, stl_name)

    # Guardar como STL
    mesh.export(stl_path)

    return stl_path


def voxelize_off(off_file, resolution=32, output_dir=None):
    """
    Voxeliza un archivo OFF y guarda como numpy array

    Args:
        off_file: Ruta al archivo OFF
        resolution: Resolución de voxelización
        output_dir: Directorio de salida (opcional)

    Returns:
        str: Ruta al archivo numpy generado
    """
    if output_dir is None:
        output_dir = os.path.dirname(off_file)

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Cargar malla OFF
    mesh = trimesh.load(off_file)

    # Centrar y normalizar malla
    mesh.vertices -= mesh.bounding_box.centroid
    scale = 1.0 / max(mesh.bounding_box.extents)
    mesh.vertices *= scale

    # Voxelizar
    voxels = mesh.voxelized(pitch=1.0 / resolution)
    voxel_grid = voxels.matrix.astype(np.float32)

    # Nombre de archivo de salida
    base_name = os.path.basename(off_file)
    npy_name = os.path.splitext(base_name)[0] + ".npy"
    npy_path = os.path.join(output_dir, npy_name)

    # Guardar como numpy array
    np.save(npy_path, voxel_grid)

    return npy_path


def process_modelnet_dataset(input_dir, output_dir, resolution=32, format="voxel"):
    """
    Procesa el dataset ModelNet que contiene archivos OFF y los convierte
    al formato especificado

    Args:
        input_dir: Directorio donde se encuentra ModelNet descomprimido
        output_dir: Directorio de salida para los archivos procesados
        resolution: Resolución para voxelización
        format: Formato de salida ('voxel' o 'stl')

    Returns:
        list: Lista de archivos procesados
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Buscar todas las carpetas de categorías
    categories = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            categories.append(item)

    if not categories:
        print(f"No se encontraron categorías en {input_dir}")
        return []

    print(f"Encontradas {len(categories)} categorías: {categories}")

    # Procesar cada categoría
    all_processed_files = []

    for category in categories:
        category_path = os.path.join(input_dir, category)

        # Crear directorio de salida para esta categoría
        category_output = os.path.join(output_dir, category)
        os.makedirs(category_output, exist_ok=True)

        # Buscar subdirectorios (train/test)
        splits = []
        for split in ["train", "test"]:
            split_path = os.path.join(category_path, split)
            if os.path.exists(split_path) and os.path.isdir(split_path):
                splits.append((split, split_path))

        # Si no hay subdirectorios train/test, procesar directamente la carpeta de categoría
        if not splits:
            splits = [("all", category_path)]

        # Procesar cada split
        for split_name, split_path in splits:
            # Crear directorio para este split
            split_output = os.path.join(category_output, split_name)
            os.makedirs(split_output, exist_ok=True)

            # Encontrar todos los archivos OFF
            off_files = []
            for file in os.listdir(split_path):
                if file.lower().endswith(".off"):
                    off_files.append(os.path.join(split_path, file))

            if not off_files:
                print(f"No se encontraron archivos OFF en {split_path}")
                continue

            print(f"Procesando {len(off_files)} archivos OFF en {category}/{split_name}")

            # Procesar cada archivo
            processed_count = 0
            for off_file in tqdm(off_files, desc=f"{category}/{split_name}"):
                try:
                    # Nombre base para archivo de salida
                    base_name = os.path.splitext(os.path.basename(off_file))[0]

                    if format == "voxel":
                        # Voxelizar
                        output_path = os.path.join(split_output, f"{base_name}.npy")

                        # Cargar malla OFF
                        mesh = trimesh.load(off_file)

                        # Centrar y normalizar
                        mesh.vertices -= mesh.bounding_box.centroid
                        scale = 0.9 / max(mesh.bounding_box.extents)  # 0.9 para dejar margen
                        mesh.vertices *= scale

                        # Voxelizar
                        voxels = mesh.voxelized(pitch=1.0 / resolution)
                        voxel_grid = voxels.matrix.astype(np.float32)

                        # Convertir a tensor compatible con redes neuronales (añadir dim de canal)
                        voxel_data = np.expand_dims(voxel_grid, axis=0)  # Añadir dimensión de canal

                        # Guardar como numpy array
                        np.save(output_path, voxel_data)

                    elif format == "stl":
                        # Convertir a STL
                        output_path = os.path.join(split_output, f"{base_name}.stl")

                        # Cargar malla OFF
                        mesh = trimesh.load(off_file)

                        # Centrar y normalizar
                        mesh.vertices -= mesh.bounding_box.centroid
                        scale = 0.9 / max(mesh.bounding_box.extents)
                        mesh.vertices *= scale

                        # Exportar como STL
                        mesh.export(output_path)

                    all_processed_files.append(output_path)
                    processed_count += 1

                except Exception as e:
                    print(f"Error procesando {off_file}: {e}")

            print(f"Procesados {processed_count}/{len(off_files)} archivos en {category}/{split_name}")

    print(f"Procesamiento completo: {len(all_processed_files)} archivos")
    return all_processed_files


def batch_convert_off_files(input_dir, output_dir=None, target_format="stl", resolution=32):
    """
    Convierte múltiples archivos OFF en un directorio

    Args:
        input_dir: Directorio con archivos OFF
        output_dir: Directorio de salida (opcional)
        target_format: Formato de salida ('stl' o 'voxel')
        resolution: Resolución para voxelización

    Returns:
        list: Lista de archivos convertidos
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, target_format)

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Encontrar todos los archivos OFF
    off_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".off"):
                off_files.append(os.path.join(root, file))

    if not off_files:
        print(f"No se encontraron archivos OFF en {input_dir}")
        return []

    print(f"Encontrados {len(off_files)} archivos OFF")

    # Procesar cada archivo
    converted_files = []
    for off_file in tqdm(off_files, desc="Convirtiendo archivos"):
        try:
            if target_format == "stl":
                output_path = convert_off_to_stl(off_file, output_dir)
                converted_files.append(output_path)
            elif target_format == "voxel":
                output_path = voxelize_off(off_file, resolution, output_dir)
                converted_files.append(output_path)
        except Exception as e:
            print(f"Error procesando {off_file}: {e}")

    print(f"Conversión completa: {len(converted_files)}/{len(off_files)} archivos")
    return converted_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar archivos OFF")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directorio con archivos OFF o dataset ModelNet')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directorio de salida')
    parser.add_argument('--resolution', type=int, default=32,
                        help='Resolución para voxelización')
    parser.add_argument('--format', type=str, default='stl', choices=['stl', 'voxel'],
                        help='Formato de salida')
    parser.add_argument('--mode', type=str, default='batch', choices=['batch', 'modelnet'],
                        help='Modo de procesamiento')

    args = parser.parse_args()

    # Configurar directorio de salida si no se especificó
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, args.format)

    # Ejecutar procesamiento según el modo
    if args.mode == 'batch':
        batch_convert_off_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_format=args.format,
            resolution=args.resolution
        )
    elif args.mode == 'modelnet':
        process_modelnet_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            resolution=args.resolution,
            format=args.format
        )