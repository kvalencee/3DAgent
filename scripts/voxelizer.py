# voxelizer.py
import os
import argparse
import json
import trimesh
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def voxelize_mesh(mesh_path, resolution=32, padding=0.05, output_dir=None, output_format='npy'):
    """
    Voxeliza un modelo 3D

    Args:
        mesh_path: Ruta al modelo
        resolution: Resolución de voxelización
        padding: Espacio adicional alrededor del modelo (0-1)
        output_dir: Directorio de salida
        output_format: Formato de salida ('npy', 'binvox', 'obj')

    Returns:
        dict: Información del proceso
    """
    try:
        # Configurar salida
        filename = os.path.basename(mesh_path)
        base_name = os.path.splitext(filename)[0]

        if output_dir is None:
            output_dir = os.path.dirname(mesh_path)

        os.makedirs(output_dir, exist_ok=True)

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

        # Guardar según formato
        output_info = {
            'input_file': mesh_path,
            'filename': filename,
            'resolution': resolution,
            'format': output_format
        }

        if output_format == 'npy':
            # Añadir dimensión de canal
            voxel_data = np.expand_dims(voxel_grid, axis=0)
            output_path = os.path.join(output_dir, f"{base_name}_{resolution}.npy")
            np.save(output_path, voxel_data)
            output_info['output_path'] = output_path

        elif output_format == 'binvox':
            # Guardar en formato binvox (requiere instalar binvox)
            output_path = os.path.join(output_dir, f"{base_name}_{resolution}.binvox")

            # Convertir a formato binvox
            import binvox_rw
            with open(output_path, 'wb') as f:
                binvox_rw.write(voxels, f)

            output_info['output_path'] = output_path

        elif output_format == 'obj':
            # Convertir voxels a malla y guardar como OBJ
            voxel_mesh = voxels.marching_cubes
            output_path = os.path.join(output_dir, f"{base_name}_voxel_{resolution}.obj")
            voxel_mesh.export(output_path)
            output_info['output_path'] = output_path

        # Estadísticas
        output_info['voxel_count'] = int(np.sum(voxel_grid > 0))
        output_info['occupancy'] = float(np.mean(voxel_grid > 0))

        return output_info

    except Exception as e:
        return {
            'input_file': mesh_path,
            'filename': os.path.basename(mesh_path),
            'error': str(e)
        }


def batch_voxelize(input_dir, output_dir=None, resolution=32, padding=0.05,
                   output_format='npy', file_types=None, max_workers=None):
    """
    Voxeliza por lotes todos los modelos 3D en un directorio

    Args:
        input_dir: Directorio de entrada
        output_dir: Directorio de salida
        resolution: Resolución de voxelización
        padding: Espacio adicional alrededor del modelo (0-1)
        output_format: Formato de salida ('npy', 'binvox', 'obj')
        file_types: Lista de extensiones de archivo a procesar
        max_workers: Número máximo de procesos simultáneos

    Returns:
        dict: Resultados del procesamiento
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                voxelize_mesh,
                model_path,
                resolution=resolution,
                padding=padding,
                output_dir=output_dir,
                output_format=output_format
            )
            for model_path in model_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Voxelizando a {resolution}³"):
            result = future.result()
            results.append(result)

    # Estadísticas
    success_count = sum(1 for r in results if 'error' not in r)
    error_count = sum(1 for r in results if 'error' in r)

    stats = {
        'total': len(model_files),
        'success': success_count,
        'errors': error_count,
        'resolution': resolution,
        'format': output_format,
        'details': results
    }

    # Guardar estadísticas
    stats_path = os.path.join(output_dir, f"voxelization_stats_{resolution}.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"Voxelización completada: {success_count} exitosos, {error_count} errores")
    print(f"Resultados guardados en: {output_dir}")
    print(f"Estadísticas guardadas en: {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Voxeliza modelos 3D por lotes")
    parser.add_argument('--input_dir', required=True, help='Directorio con modelos 3D')
    parser.add_argument('--output_dir', help='Directorio de salida')
    parser.add_argument('--resolution', type=int, default=32, help='Resolución de voxelización')
    parser.add_argument('--padding', type=float, default=0.05, help='Espacio adicional (0-1)')
    parser.add_argument('--format', choices=['npy', 'binvox', 'obj'], default='npy',
                        help='Formato de salida')
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
        output_format=args.format,
        file_types=file_types,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()