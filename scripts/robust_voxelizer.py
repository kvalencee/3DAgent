# robust_voxelizer.py
import os
import argparse
import json
import trimesh
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def voxelize_mesh(mesh_path, resolution=64, padding=0.05, output_dir=None):
    """
    Voxeliza un modelo 3D con manejo robusto de errores
    """
    try:
        # Configurar salida
        filename = os.path.basename(mesh_path)
        base_name = os.path.splitext(filename)[0]

        if output_dir is None:
            output_dir = os.path.dirname(mesh_path)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_{resolution}.npy")

        # Cargar malla
        mesh = trimesh.load(mesh_path)

        # Normalizar la malla
        mesh.vertices -= mesh.bounding_box.centroid  # Centrar

        # Aplicar padding
        scale = (1.0 - padding) / max(mesh.bounding_box.extents)
        mesh.apply_scale(scale)

        # Voxelizar
        voxels = mesh.voxelized(pitch=1.0 / resolution)
        voxel_grid = voxels.matrix.astype(np.float32)

        # Asegurarse de que la dimensión es correcta
        if voxel_grid.shape != (resolution, resolution, resolution):
            # Redimensionar o rellenar
            old_shape = voxel_grid.shape
            new_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

            # Copiar datos existentes
            min_x = min(old_shape[0], resolution)
            min_y = min(old_shape[1], resolution)
            min_z = min(old_shape[2], resolution)

            new_grid[:min_x, :min_y, :min_z] = voxel_grid[:min_x, :min_y, :min_z]
            voxel_grid = new_grid

        # Añadir dimensión de canal
        voxel_data = np.expand_dims(voxel_grid, axis=0)

        # Guardar como numpy array
        np.save(output_path, voxel_data)

        # Estadísticas
        output_info = {
            'input_file': mesh_path,
            'filename': filename,
            'resolution': resolution,
            'voxel_count': int(np.sum(voxel_grid > 0)),
            'occupancy': float(np.mean(voxel_grid > 0)),
            'output_path': output_path,
            'status': 'success'
        }

        return output_info

    except Exception as e:
        return {
            'input_file': mesh_path,
            'filename': os.path.basename(mesh_path),
            'error': str(e),
            'status': 'failed'
        }


def batch_voxelize(input_dir, output_dir=None, resolution=64, padding=0.05, file_types=None, max_workers=None):
    """
    Voxeliza por lotes todos los modelos 3D en un directorio con manejo robusto de errores
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, f"voxels_{resolution}")

    os.makedirs(output_dir, exist_ok=True)

    # Extensiones de archivo por defecto
    if file_types is None:
        file_types = ['.stl', '.obj', '.ply']

    # Buscar archivos
    model_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_types):
                model_files.append(os.path.join(root, file))

    print(f"Encontrados {len(model_files)} archivos para voxelizar")

    # Configurar número de trabajadores
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    # Voxelizar en paralelo
    results = []
    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for model_path in model_files:
            future = executor.submit(
                voxelize_mesh,
                model_path,
                resolution=resolution,
                padding=padding,
                output_dir=output_dir
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Voxelizando a {resolution}³"):
            try:
                result = future.result()
                results.append(result)

                if result['status'] == 'success':
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"Error en procesamiento: {str(e)}")
                error_count += 1

    # Estadísticas
    stats = {
        'total': len(model_files),
        'success': success_count,
        'errors': error_count,
        'resolution': resolution,
        'details': results
    }

    # Guardar estadísticas
    stats_path = os.path.join(output_dir, f"voxelization_stats_{resolution}.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Guardar lista de archivos fallidos
    failed_results = [r for r in results if r.get('status') == 'failed']
    failed_path = os.path.join(output_dir, f"failed_voxelization_{resolution}.txt")
    with open(failed_path, 'w', encoding='utf-8') as f:
        for result in failed_results:
            f.write(f"{result.get('filename')}: {result.get('error')}\n")

    print(f"Voxelización completada: {success_count} exitosos, {error_count} errores")
    print(f"Resultados guardados en: {output_dir}")
    print(f"Estadísticas guardadas en: {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Voxeliza modelos 3D por lotes")
    parser.add_argument('--input_dir', required=True, help='Directorio con modelos 3D')
    parser.add_argument('--output_dir', help='Directorio de salida')
    parser.add_argument('--resolution', type=int, default=64, help='Resolución de voxelización')
    parser.add_argument('--padding', type=float, default=0.05, help='Espacio adicional (0-1)')
    parser.add_argument('--file_types', default='.stl,.obj,.ply',
                        help='Extensiones de archivo separadas por coma')
    parser.add_argument('--workers', type=int, help='Número de procesos simultáneos')

    args = parser.parse_args()

    file_types = [ext.strip() for ext in args.file_types.split(',')]

    batch_voxelize(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        padding=args.padding,
        file_types=file_types,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()