# prepare_training_data.py
import os
import argparse
import json
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def prepare_training_data(voxel_dir, metadata_file, output_dir, train_ratio=0.8, val_ratio=0.1,
                          test_ratio=0.1, balanced=False, category_field='type', min_samples=3):
    """
    Prepara datos para entrenamiento con manejo robusto
    """
    # Verificar proporciones
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"

    # Crear directorios
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Cargar metadatos
    metadata = {}
    try:
        if metadata_file.lower().endswith('.csv'):
            metadata_df = pd.read_csv(metadata_file)
            # Convertir a diccionario para acceso más fácil
            for _, row in metadata_df.iterrows():
                try:
                    filename = row['filename']
                    metadata[filename] = row.to_dict()
                except:
                    continue
        elif metadata_file.lower().endswith('.json'):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            print(f"Formato de metadatos no compatible: {metadata_file}")
            return {}
    except Exception as e:
        print(f"Error cargando metadatos: {str(e)}")
        print("Continuando sin metadatos completos...")

    print(f"Metadatos cargados para {len(metadata)} modelos")

    # Buscar archivos voxel
    voxel_files = []
    for root, _, files in os.walk(voxel_dir):
        for file in files:
            if file.lower().endswith('.npy'):
                voxel_files.append((os.path.join(root, file), file))

    print(f"Encontrados {len(voxel_files)} archivos de voxels")

    # Filtrar por modelos con metadatos si se requiere balanceo
    if balanced and category_field and metadata:
        voxel_files_with_metadata = []

        for file_path, filename in voxel_files:
            # Buscar nombre base
            base_name = os.path.splitext(filename)[0]

            # Buscar correspondencia en metadatos
            found = False
            for meta_filename in metadata:
                if base_name.startswith(os.path.splitext(meta_filename)[0]):
                    voxel_files_with_metadata.append((file_path, filename, meta_filename))
                    found = True
                    break

            if not found:
                # Si no hay metadatos, asumir categoría desconocida
                voxel_files_with_metadata.append((file_path, filename, None))

        voxel_files = voxel_files_with_metadata

    # Dividir por categoría si es necesario
    if balanced and category_field and metadata:
        # Organizar por categoría
        categories = {}

        for file_info in voxel_files:
            if len(file_info) == 3:
                file_path, filename, meta_filename = file_info
            else:
                file_path, filename = file_info
                meta_filename = None

            # Determinar categoría
            category = 'unknown'
            if meta_filename and meta_filename in metadata:
                category = metadata[meta_filename].get(category_field, 'unknown')

            if category not in categories:
                categories[category] = []

            categories[category].append((file_path, filename))

        # Dividir cada categoría
        train_files = []
        val_files = []
        test_files = []

        for category, files in categories.items():
            if category == 'unknown':
                # Asignar desconocidos principalmente a entrenamiento
                if len(files) >= min_samples:
                    train_size = int(len(files) * 0.8)
                    val_size = int(len(files) * 0.1)
                    test_size = len(files) - train_size - val_size

                    train_files.extend(files[:train_size])
                    val_files.extend(files[train_size:train_size + val_size])
                    test_files.extend(files[train_size + val_size:])
                else:
                    # Si hay muy pocos, todos a entrenamiento
                    train_files.extend(files)

                continue

            print(f"Categoría '{category}': {len(files)} archivos")

            if len(files) < min_samples:
                print(f"  Advertencia: Muy pocos archivos ({len(files)}) para dividir correctamente")
                if len(files) > 0:
                    train_files.extend(files)  # Todos a entrenamiento
                continue

            # Dividir
            train_val, test_cat = train_test_split(files, test_size=test_ratio, random_state=42)
            train_cat, val_cat = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio),
                                                  random_state=42)

            train_files.extend(train_cat)
            val_files.extend(val_cat)
            test_files.extend(test_cat)

            print(f"  Train: {len(train_cat)}, Val: {len(val_cat)}, Test: {len(test_cat)}")
    else:
        # Simplificar lista si es necesario
        if len(voxel_files) > 0 and len(voxel_files[0]) == 3:
            voxel_files = [(file_path, filename) for file_path, filename, _ in voxel_files]

        # División simple
        train_val, test_files = train_test_split(voxel_files, test_size=test_ratio, random_state=42)
        train_files, val_files = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio),
                                                  random_state=42)

    # Preparar cada conjunto
    sets = [
        ('train', train_files, train_dir),
        ('val', val_files, val_dir),
        ('test', test_files, test_dir)
    ]

    stats = {
        'total': len(voxel_files),
        'sets': {}
    }

    # Procesamiento para cada conjunto
    for set_name, files, set_dir in sets:
        print(f"Procesando conjunto {set_name} ({len(files)} archivos)")

        set_stats = {
            'count': len(files),
            'categories': {}
        }

        # Crear estructura de directorios por categoría
        if category_field and metadata:
            categories_seen = set()
            for meta_filename, meta_data in metadata.items():
                if category_field in meta_data:
                    category = meta_data[category_field]
                    if category and category != 'unknown':
                        categories_seen.add(category)

            for category in categories_seen:
                os.makedirs(os.path.join(set_dir, category), exist_ok=True)

        # Copiar archivos
        for file_info in tqdm(files, desc=f"Copiando {set_name}"):
            file_path, filename = file_info

            # Buscar categoría si hay metadatos
            category = 'unknown'
            if metadata:
                base_name = os.path.splitext(filename)[0]

                # Buscar correspondencia en metadatos
                for meta_filename, meta_data in metadata.items():
                    if base_name.startswith(os.path.splitext(meta_filename)[0]):
                        if category_field in meta_data:
                            category = meta_data[category_field]
                        break

            if category not in set_stats['categories']:
                set_stats['categories'][category] = 0

            set_stats['categories'][category] += 1

            # Determinar directorio destino
            if category_field and category != 'unknown':
                dest_dir = os.path.join(set_dir, category)
                os.makedirs(dest_dir, exist_ok=True)
            else:
                dest_dir = set_dir

            # Copiar archivo
            try:
                dest_path = os.path.join(dest_dir, filename)
                shutil.copy2(file_path, dest_path)
            except Exception as e:
                print(f"Error copiando {file_path} a {dest_path}: {str(e)}")

        stats['sets'][set_name] = set_stats

    # Guardar estadísticas
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"Preparación completada. Resultados guardados en {output_dir}")
    print(f"Estadísticas guardadas en {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepara datos para entrenamiento")
    parser.add_argument('--voxel_dir', required=True, help='Directorio con archivos de voxels')
    parser.add_argument('--metadata_file', required=True, help='Archivo de metadatos')
    parser.add_argument('--output_dir', required=True, help='Directorio de salida')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Proporción para entrenamiento')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Proporción para validación')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Proporción para prueba')
    parser.add_argument('--balanced', action='store_true', help='Balancear por categoría')
    parser.add_argument('--category_field', default='type', help='Campo de categoría en metadatos')
    parser.add_argument('--min_samples', type=int, default=3, help='Mínimo de muestras por categoría para dividir')

    args = parser.parse_args()

    prepare_training_data(
        voxel_dir=args.voxel_dir,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        balanced=args.balanced,
        category_field=args.category_field,
        min_samples=args.min_samples
    )


if __name__ == "__main__":
    main()