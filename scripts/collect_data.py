# collect_data.py
import os
import argparse
import json
from tqdm import tqdm

# Importar funciones desde los otros módulos
from thingiverse_downloader import download_from_thingiverse
from sketchfab_downloader import download_from_sketchfab
from academic_downloader import download_academic_dataset
from off_processor import process_modelnet_dataset, batch_convert_off_files


def collect_data(args):
    """
    Función principal para recopilar datos de diferentes fuentes

    Args:
        args: Argumentos de línea de comandos
    """
    # Registro de estadísticas
    stats = {
        "sources": {},
        "processed_files": 0,
        "total_models": 0
    }

    # Descargar de fuentes seleccionadas
    if args.thingiverse and args.thingiverse_api_key:
        print("\n=== Descargando modelos de Thingiverse ===")
        thingiverse_dir = os.path.join(args.output_dir, "raw", "thingiverse")
        models_count = download_from_thingiverse(
            api_key=args.thingiverse_api_key,
            category=args.thingiverse_category,
            keyword=args.thingiverse_keyword,
            min_likes=args.thingiverse_min_likes,
            max_models=args.max_models,
            output_dir=thingiverse_dir
        )
        stats["sources"]["thingiverse"] = models_count
        stats["total_models"] += models_count

    if args.sketchfab and args.sketchfab_api_key:
        print("\n=== Descargando modelos de Sketchfab ===")
        sketchfab_dir = os.path.join(args.output_dir, "raw", "sketchfab")
        models_count = download_from_sketchfab(
            api_key=args.sketchfab_api_key,
            category=args.sketchfab_category,
            count=args.max_models,
            output_dir=sketchfab_dir
        )
        stats["sources"]["sketchfab"] = models_count
        stats["total_models"] += models_count

    if args.academic:
        print("\n=== Descargando datasets académicos ===")
        academic_dir = os.path.join(args.output_dir, "raw", "academic")

        for dataset in args.academic_datasets.split(","):
            dataset = dataset.strip()
            if dataset:
                files_count = download_academic_dataset(
                    name=dataset,
                    category=args.academic_category,
                    output_dir=os.path.join(academic_dir, dataset)
                )
                stats["sources"][dataset] = files_count
                stats["total_models"] += files_count

    # Procesar datos descargados
    if args.process:
        print("\n=== Procesando archivos descargados ===")
        processed_dir = os.path.join(args.output_dir, "processed")

        # Procesar ModelNet (archivos OFF)
        if args.academic:
            academic_dir = os.path.join(args.output_dir, "raw", "academic")

            for dataset in os.listdir(academic_dir):
                dataset_path = os.path.join(academic_dir, dataset)
                if os.path.isdir(dataset_path) and (dataset.startswith("ModelNet") or "off" in dataset.lower()):
                    print(f"\nProcesando archivos OFF de {dataset}...")
                    output_format_dir = os.path.join(processed_dir, args.format)
                    processed_files = process_modelnet_dataset(
                        input_dir=dataset_path,
                        output_dir=os.path.join(output_format_dir, dataset),
                        resolution=args.resolution,
                        format=args.format
                    )
                    stats["processed_files"] += len(processed_files)

        # Procesar archivos STL/OBJ de otras fuentes a voxels
        if args.voxelize_all and args.format == "voxel":
            sources = []
            if args.thingiverse:
                sources.append(os.path.join(args.output_dir, "raw", "thingiverse"))
            if args.sketchfab:
                sources.append(os.path.join(args.output_dir, "raw", "sketchfab"))

            if sources:
                print("\nVoxelizando archivos 3D de otras fuentes...")
                for source_dir in sources:
                    for root, dirs, files in os.walk(source_dir):
                        mesh_files = [f for f in files if f.lower().endswith((".stl", ".obj", ".ply"))]
                        if mesh_files:
                            relative_path = os.path.relpath(root, args.output_dir)
                            output_path = os.path.join(processed_dir, "voxels", relative_path)

                            # Crear estructura de directorios para preservar jerarquía
                            os.makedirs(output_path, exist_ok=True)

                            for mesh_file in tqdm(mesh_files, desc=f"Voxelizando en {relative_path}"):
                                try:
                                    import trimesh
                                    import numpy as np

                                    # Cargar malla
                                    mesh_path = os.path.join(root, mesh_file)
                                    mesh = trimesh.load(mesh_path)

                                    # Centrar y normalizar
                                    mesh.vertices -= mesh.bounding_box.centroid
                                    scale = 0.9 / max(mesh.bounding_box.extents)
                                    mesh.vertices *= scale

                                    # Voxelizar
                                    voxels = mesh.voxelized(pitch=1.0 / args.resolution)
                                    voxel_grid = voxels.matrix.astype(np.float32)

                                    # Añadir dimensión de canal
                                    voxel_data = np.expand_dims(voxel_grid, axis=0)

                                    # Guardar como numpy array
                                    base_name = os.path.splitext(mesh_file)[0]
                                    output_file = os.path.join(output_path, f"{base_name}.npy")
                                    np.save(output_file, voxel_data)

                                    stats["processed_files"] += 1

                                except Exception as e:
                                    print(f"Error procesando {mesh_file}: {e}")

    # Guardar estadísticas
    stats_path = os.path.join(args.output_dir, "data_collection_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n=== Resumen de recopilación de datos ===")
    print(f"Total de modelos descargados: {stats['total_models']}")
    print(f"Total de archivos procesados: {stats['processed_files']}")
    print(f"Estadísticas guardadas en: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Recopilar y procesar datos para entrenamiento de modelos 3D")

    # Opciones generales
    parser.add_argument('--output_dir', type=str, default="data",
                        help='Directorio base de salida')
    parser.add_argument('--max_models', type=int, default=50,
                        help='Número máximo de modelos a descargar por fuente')
    parser.add_argument('--resolution', type=int, default=32,
                        help='Resolución para voxelización')
    parser.add_argument('--format', type=str, default="voxel", choices=["voxel", "stl"],
                        help='Formato de procesamiento')

    # Opciones de fuentes
    parser.add_argument('--thingiverse', action='store_true',
                        help='Descargar de Thingiverse')
    parser.add_argument('--sketchfab', action='store_true',
                        help='Descargar de Sketchfab')
    parser.add_argument('--academic', action='store_true',
                        help='Descargar datasets académicos')

    # Opciones de procesamiento
    parser.add_argument('--process', action='store_true',
                        help='Procesar archivos después de descargar')
    parser.add_argument('--voxelize_all', action='store_true',
                        help='Voxelizar todos los archivos 3D encontrados')

    # Opciones de Thingiverse
    parser.add_argument('--thingiverse_api_key', type=str, default=None,
                        help='Clave API de Thingiverse')
    parser.add_argument('--thingiverse_category', type=str, default="home",
                        help='Categoría de Thingiverse')
    parser.add_argument('--thingiverse_keyword', type=str, default="decor",
                        help='Palabra clave para búsqueda en Thingiverse')
    parser.add_argument('--thingiverse_min_likes', type=int, default=10,
                        help='Mínimo de likes para modelos de Thingiverse')

    # Opciones de Sketchfab
    parser.add_argument('--sketchfab_api_key', type=str, default=None,
                        help='Clave API de Sketchfab')
    parser.add_argument('--sketchfab_category', type=str, default="furniture",
                        help='Categoría de Sketchfab')

    # Opciones de datasets académicos
    parser.add_argument('--academic_datasets', type=str, default="ModelNet10",
                        help='Datasets a descargar (separados por coma)')
    parser.add_argument('--academic_category', type=str, default=None,
                        help='Categoría específica a extraer (opcional)')

    args = parser.parse_args()

    # Validar argumentos
    if not any([args.thingiverse, args.sketchfab, args.academic]):
        print("Error: Debe seleccionar al menos una fuente de datos (--thingiverse, --sketchfab, --academic)")
        parser.print_help()
        return

    if args.thingiverse and not args.thingiverse_api_key:
        print("Advertencia: Se requiere una clave API para Thingiverse")

    if args.sketchfab and not args.sketchfab_api_key:
        print("Advertencia: Se requiere una clave API para Sketchfab")

    # Ejecutar recopilación de datos
    collect_data(args)


if __name__ == "__main__":
    main()