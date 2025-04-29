# data_preparation.py
import os
import argparse
import numpy as np
import trimesh
import random
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def normalize_dataset(input_dir, output_dir, target_resolution=32, target_format="voxel", test_split=0.2):
    """
    Normaliza un conjunto de datos, convirtiendo a un formato uniforme y dividiendo en train/test

    Args:
        input_dir: Directorio con archivos 3D
        output_dir: Directorio de salida
        target_resolution: Resolución para voxelización
        target_format: Formato de salida ('voxel' o 'mesh')
        test_split: Proporción para conjunto de prueba

    Returns:
        dict: Estadísticas del proceso
    """
    os.makedirs(output_dir, exist_ok=True)

    # Crear directorios para train/test
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Estadísticas
    stats = {
        "total_files": 0,
        "processed_files": 0,
        "train_files": 0,
        "test_files": 0,
        "categories": {}
    }

    # Buscar todas las categorías
    categories = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            categories.append(item)

    if not categories:
        # Si no hay subdirectorios, tratar como una sola categoría
        categories = [""]

    print(f"Procesando {len(categories)} categorías: {categories}")

    # Procesar cada categoría
    for category in categories:
        category_path = os.path.join(input_dir, category) if category else input_dir
        category_name = category or "uncategorized"

        # Crear directorios de salida para esta categoría
        train_category_dir = os.path.join(train_dir, category_name)
        test_category_dir = os.path.join(test_dir, category_name)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)

        # Buscar archivos 3D
        all_files = []
        for root, _, files in os.walk(category_path):
            for file in files:
                if file.lower().endswith((".stl", ".obj", ".ply", ".off", ".npy")):
                    all_files.append(os.path.join(root, file))

        if not all_files:
            print(f"No se encontraron archivos 3D en {category_path}")
            continue

        stats["total_files"] += len(all_files)
        stats["categories"][category_name] = len(all_files)

        # Dividir en train/test
        train_files, test_files = train_test_split(all_files, test_size=test_split, random_state=42)

        stats["train_files"] += len(train_files)
        stats["test_files"] += len(test_files)

        # Procesar archivos de entrenamiento
        for file_path in tqdm(train_files, desc=f"Procesando {category_name} (train)"):
            try:
                process_file(file_path, train_category_dir, target_resolution, target_format)
                stats["processed_files"] += 1
            except Exception as e:
                print(f"Error procesando {file_path}: {e}")

        # Procesar archivos de prueba
        for file_path in tqdm(test_files, desc=f"Procesando {category_name} (test)"):
            try:
                process_file(file_path, test_category_dir, target_resolution, target_format)
                stats["processed_files"] += 1
            except Exception as e:
                print(f"Error procesando {file_path}: {e}")

    print(f"Preparación de datos completada:")
    print(f"  Total de archivos encontrados: {stats['total_files']}")
    print(f"  Archivos procesados: {stats['processed_files']}")
    print(f"  Conjunto de entrenamiento: {stats['train_files']} archivos")
    print(f"  Conjunto de prueba: {stats['test_files']} archivos")

    return stats


def process_file(file_path, output_dir, resolution=32, target_format="voxel"):
    """
    Procesa un archivo 3D al formato objetivo

    Args:
        file_path: Ruta al archivo 3D
        output_dir: Directorio de salida
        resolution: Resolución para voxelización
        target_format: Formato de salida ('voxel' o 'mesh')

    Returns:
        str: Ruta al archivo procesado
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    if target_format == "voxel":
        # Convertir a voxel (npy)
        output_path = os.path.join(output_dir, f"{base_name}.npy")

        if file_ext == ".npy":
            # Ya es un archivo numpy, verificar formato
            voxels = np.load(file_path)

            # Normalizar formato
            if voxels.ndim < 3:
                raise ValueError(f"Archivo numpy con dimensión incorrecta: {voxels.ndim}")

            # Si tiene más de 3 dimensiones (batch o canal), extraer
            while voxels.ndim > 3:
                voxels = voxels[0]

            # Cambiar resolución si es necesario
            current_res = voxels.shape[0]
            if current_res != resolution:
                from scipy.ndimage import zoom
                zoom_factor = resolution / current_res
                voxels = zoom(voxels, zoom_factor, order=1)

            # Añadir dimensión de canal
            voxels = np.expand_dims(voxels, axis=0)

            # Guardar
            np.save(output_path, voxels)

        else:
            # Cargar malla
            mesh = trimesh.load(file_path)

            # Centrar y normalizar
            mesh.vertices -= mesh.bounding_box.centroid
            scale = 0.9 / max(mesh.bounding_box.extents)
            mesh.vertices *= scale

            # Voxelizar
            voxels = mesh.voxelized(pitch=1.0 / resolution)
            voxel_grid = voxels.matrix.astype(np.float32)

            # Añadir dimensión de canal
            voxel_data = np.expand_dims(voxel_grid, axis=0)

            # Guardar
            np.save(output_path, voxel_data)

    elif target_format == "mesh":
        # Convertir a malla (stl)
        output_path = os.path.join(output_dir, f"{base_name}.stl")

        if file_ext == ".npy":
            # Convertir de voxel a malla
            import mcubes

            # Cargar voxels
            voxels = np.load(file_path)

            # Si tiene dimensión de batch o canal, extraer
            while voxels.ndim > 3:
                voxels = voxels[0]

            # Aplicar umbral
            voxels_binary = voxels > 0.5

            # Cambiar resolución si es necesario
            current_res = voxels.shape[0]
            if current_res != resolution:
                from scipy.ndimage import zoom
                zoom_factor = resolution / current_res
                voxels_binary = zoom(voxels_binary, zoom_factor, order=0)

            # Aplicar marching cubes
            vertices, triangles = mcubes.marching_cubes(voxels_binary, 0)

            # Normalizar vértices al rango [-0.5, 0.5]
            vertices = vertices / resolution - 0.5

            # Crear malla
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

            # Exportar
            mesh.export(output_path)

        else:
            # Cargar malla
            mesh = trimesh.load(file_path)

            # Centrar y normalizar
            mesh.vertices -= mesh.bounding_box.centroid
            scale = 0.9 / max(mesh.bounding_box.extents)
            mesh.vertices *= scale

            # Exportar
            mesh.export(output_path)

    return output_path


def create_balanced_dataset(input_dir, output_dir, max_per_category=None, min_per_category=None, augment=False):
    """
    Crea un conjunto de datos balanceado a partir de un directorio de entrada

    Args:
        input_dir: Directorio con datos
        output_dir: Directorio de salida
        max_per_category: Número máximo de ejemplos por categoría
        min_per_category: Número mínimo de ejemplos por categoría
        augment: Realizar aumento de datos para categorías con pocos ejemplos

    Returns:
        dict: Estadísticas del proceso
    """
    os.makedirs(output_dir, exist_ok=True)

    # Estadísticas
    stats = {
        "original_counts": {},
        "final_counts": {},
        "augmented": 0
    }

    # Contar archivos por categoría
    categories = {}
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.npy') or f.endswith('.stl')]
            if files:
                categories[category] = files
                stats["original_counts"][category] = len(files)

    if not categories:
        print(f"No se encontraron categorías válidas en {input_dir}")
        return stats

    print(f"Encontradas {len(categories)} categorías")

    # Determinar tamaño objetivo por categoría
    if max_per_category is None:
        # Usar el mínimo de todas las categorías
        counts = [len(files) for files in categories.values()]
        max_per_category = min(counts) if min_per_category is None else max(min(counts), min_per_category)

    # Procesar cada categoría
    for category, files in categories.items():
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)

        if len(files) <= max_per_category:
            # Copiar todos los archivos
            target_count = len(files)
            selected_files = files
        else:
            # Seleccionar subset aleatorio
            target_count = max_per_category
            selected_files = random.sample(files, target_count)

        # Copiar archivos seleccionados
        for file in tqdm(selected_files, desc=f"Procesando {category}"):
            src_path = os.path.join(input_dir, category, file)
            dst_path = os.path.join(category_output_dir, file)
            shutil.copy2(src_path, dst_path)

        # Aumentar datos si es necesario
        if augment and min_per_category and len(files) < min_per_category:
            augmented_count = 0
            needed = min_per_category - len(files)

            for i in range(needed):
                # Seleccionar archivo aleatorio
                src_file = random.choice(files)
                src_path = os.path.join(input_dir, category, src_file)

                # Crear versión aumentada
                base_name = os.path.splitext(src_file)[0]
                ext = os.path.splitext(src_file)[1]
                dst_path = os.path.join(category_output_dir, f"{base_name}_aug_{i + 1}{ext}")

                if ext.lower() == '.npy':
                    # Aumentar datos de voxel
                    try:
                        voxels = np.load(src_path)

                        # Aplicar transformaciones aleatorias
                        if random.random() > 0.5:
                            # Rotación 90 grados aleatoria
                            k = random.randint(1, 3)
                            voxels = np.rot90(voxels, k=k, axes=(1, 2))

                        if random.random() > 0.5:
                            # Volteo horizontal
                            voxels = np.flip(voxels, axis=2)

                        if random.random() > 0.5:
                            # Volteo vertical
                            voxels = np.flip(voxels, axis=1)

                        # Guardar aumentado
                        np.save(dst_path, voxels)
                        augmented_count += 1

                    except Exception as e:
                        print(f"Error aumentando {src_path}: {e}")

                elif ext.lower() in ['.stl', '.obj', '.ply']:
                    # Aumentar datos de malla
                    try:
                        mesh = trimesh.load(src_path)

                        # Aplicar transformaciones aleatorias
                        if random.random() > 0.5:
                            # Rotación aleatoria
                            angle = random.uniform(0, 2 * np.pi)
                            axis = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                            mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))

                        if random.random() > 0.5:
                            # Escala ligeramente diferente
                            scale = random.uniform(0.9, 1.1)
                            mesh.apply_scale(scale)

                        # Exportar aumentado
                        mesh.export(dst_path)
                        augmented_count += 1

                    except Exception as e:
                        print(f"Error aumentando {src_path}: {e}")

            stats["augmented"] += augmented_count

        # Contar archivos finales
        final_files = os.listdir(category_output_dir)
        stats["final_counts"][category] = len(final_files)

    print(f"Creación de conjunto balanceado completada:")
    for category in categories:
        print(f"  {category}: {stats['original_counts'][category]} → {stats['final_counts'][category]}")
    if stats["augmented"] > 0:
        print(f"  Total aumentado: {stats['augmented']} ejemplos")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preparar datos para entrenamiento de modelos 3D")

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # Comando normalize
    normalize_parser = subparsers.add_parser("normalize", help="Normalizar conjunto de datos")
    normalize_parser.add_argument('--input_dir', type=str, required=True,
                                  help='Directorio con datos a normalizar')
    normalize_parser.add_argument('--output_dir', type=str, required=True,
                                  help='Directorio de salida')
    normalize_parser.add_argument('--resolution', type=int, default=32,
                                  help='Resolución para voxelización')
    normalize_parser.add_argument('--format', type=str, choices=['voxel', 'mesh'], default='voxel',
                                  help='Formato de salida')
    normalize_parser.add_argument('--test_split', type=float, default=0.2,
                                  help='Proporción para conjunto de prueba')

    # Comando balance
    balance_parser = subparsers.add_parser("balance", help="Balancear conjunto de datos")
    balance_parser.add_argument('--input_dir', type=str, required=True,
                                help='Directorio con datos a balancear')
    balance_parser.add_argument('--output_dir', type=str, required=True,
                                help='Directorio de salida')
    balance_parser.add_argument('--max_per_category', type=int, default=None,
                                help='Número máximo de ejemplos por categoría')
    balance_parser.add_argument('--min_per_category', type=int, default=None,
                                help='Número mínimo de ejemplos por categoría')
    balance_parser.add_argument('--augment', action='store_true',
                                help='Realizar aumento de datos para categorías con pocos ejemplos')

    args = parser.parse_args()

    if args.command == "normalize":
        normalize_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_resolution=args.resolution,
            target_format=args.format,
            test_split=args.test_split
        )
    elif args.command == "balance":
        create_balanced_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_per_category=args.max_per_category,
            min_per_category=args.min_per_category,
            augment=args.augment
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()