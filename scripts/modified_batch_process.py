# modified_batch_process.py

import os
import argparse
import json
import trimesh
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def normalize_model(input_path, output_path=None, scale=True, center=True, repair=True):
    """
    Normaliza un modelo 3D con manejo robusto de errores
    """
    # Crear ruta de salida si no se proporciona
    if output_path is None:
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(dirname, f"{name}_normalized{ext}")

    try:
        # Cargar modelo
        mesh = trimesh.load(input_path)

        # Centrar en el origen
        if center:
            mesh.vertices -= mesh.bounding_box.centroid

        # Escalar a tamaño unitario
        if scale:
            scale_factor = 1.0 / max(mesh.bounding_box.extents)
            mesh.apply_scale(scale_factor)

        # Reparar problemas comunes
        if repair:
            if hasattr(mesh, 'process'):
                mesh.process(validate=True)
            # Reparación básica - asegurar que las caras tienen orientación coherente
            try:
                mesh.fix_normals()
            except (AttributeError, ValueError):
                # Si no puede arreglar normales, intentamos continuar
                pass

        # Guardar modelo normalizado
        mesh.export(output_path)

        # Recopilar propiedades básicas sin depender de atributos específicos
        props = {
            'filename': os.path.basename(input_path),
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'is_watertight': getattr(mesh, 'is_watertight', False),
            'dimensions': mesh.bounding_box.extents.tolist()
        }

        # Propiedades adicionales solo si están disponibles
        if hasattr(mesh, 'is_manifold'):
            props['is_manifold'] = mesh.is_manifold

        # Calcular volumen si es estanco
        if getattr(mesh, 'is_watertight', False):
            try:
                props['volume'] = float(mesh.volume)
            except (AttributeError, ValueError):
                props['volume'] = None

        return mesh, output_path, props

    except Exception as e:
        return None, None, {
            'filename': os.path.basename(input_path),
            'error': str(e),
            'status': 'failed'
        }


def process_model(input_path, output_dir, analyze_only=False):
    """
    Procesa un modelo 3D con manejo de errores mejorado
    """
    try:
        # Crear ruta de salida
        basename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, basename)

        if not analyze_only:
            # Normalizar modelo
            mesh, saved_path, props = normalize_model(input_path, output_path)
            if mesh is None:
                return props  # Ya contiene información de error
        else:
            # Solo cargar modelo
            try:
                mesh = trimesh.load(input_path)
                # Analizar modelo
                props = {
                    'filename': basename,
                    'filepath': input_path,
                    'vertices': len(mesh.vertices),
                    'faces': len(mesh.faces),
                    'is_watertight': getattr(mesh, 'is_watertight', False),
                    'dimensions': mesh.bounding_box.extents.tolist(),
                    'status': 'analyzed'
                }

                # Propiedades adicionales solo si están disponibles
                if hasattr(mesh, 'is_manifold'):
                    props['is_manifold'] = mesh.is_manifold

                # Calcular volumen si es estanco
                if getattr(mesh, 'is_watertight', False):
                    try:
                        props['volume'] = float(mesh.volume)
                    except (AttributeError, ValueError):
                        props['volume'] = None

                return props
            except Exception as e:
                return {
                    'filename': basename,
                    'filepath': input_path,
                    'error': str(e),
                    'status': 'analysis_failed'
                }

        # Si llegamos aquí, el procesamiento fue exitoso
        props['status'] = 'processed'
        props['output_path'] = saved_path
        return props

    except Exception as e:
        return {
            'filename': os.path.basename(input_path),
            'filepath': input_path,
            'error': str(e),
            'status': 'processing_failed'
        }


def batch_process_models(input_dir, output_dir=None, normalize=True, max_workers=4, file_types=None):
    """
    Procesa por lotes todos los modelos 3D en un directorio con manejo de errores mejorado
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "normalized")

    # Crear directorio de salida
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

    print(f"Encontrados {len(model_files)} archivos para procesar")

    # Procesar archivos en paralelo
    results = []
    processed_count = 0
    error_count = 0

    # Usar método secuencial para depuración o parallel para producción
    use_parallel = True

    if use_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_model, model_path, output_dir, not normalize)
                for model_path in model_files
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando modelos"):
                result = future.result()
                results.append(result)

                if result.get('status') in ['failed', 'analysis_failed', 'processing_failed']:
                    error_count += 1
                else:
                    processed_count += 1
    else:
        # Método secuencial para depuración
        for model_path in tqdm(model_files, desc="Procesando modelos"):
            result = process_model(model_path, output_dir, not normalize)
            results.append(result)

            if result.get('status') in ['failed', 'analysis_failed', 'processing_failed']:
                error_count += 1
                print(f"Error: {result.get('error')} en {result.get('filename')}")
            else:
                processed_count += 1

    # Separar resultados exitosos y fallidos
    successful_results = [r for r in results if
                          r.get('status') not in ['failed', 'analysis_failed', 'processing_failed']]
    failed_results = [r for r in results if r.get('status') in ['failed', 'analysis_failed', 'processing_failed']]

    # Guardar resultados
    stats = {
        'total': len(model_files),
        'processed': processed_count,
        'errors': error_count,
        'details': {
            'successful': successful_results,
            'failed': failed_results
        }
    }

    stats_path = os.path.join(output_dir, "processing_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Guardar lista de archivos fallidos para referencia
    failed_path = os.path.join(output_dir, "failed_models.txt")
    with open(failed_path, 'w', encoding='utf-8') as f:
        for result in failed_results:
            f.write(f"{result.get('filename')}: {result.get('error')}\n")

    print(f"Procesamiento completado: {processed_count} exitosos, {error_count} errores")
    print(f"Resultados guardados en: {stats_path}")
    print(f"Lista de archivos fallidos: {failed_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Procesa por lotes modelos 3D")
    parser.add_argument('--input_dir', required=True, help='Directorio con modelos 3D')
    parser.add_argument('--output_dir', help='Directorio de salida (por defecto: input_dir/normalized)')
    parser.add_argument('--no_normalize', action='store_true', help='Solo analizar, sin normalizar')
    parser.add_argument('--workers', type=int, default=4, help='Número de procesos simultáneos')
    parser.add_argument('--file_types', default='.stl,.obj,.ply', help='Extensiones de archivo separadas por coma')

    args = parser.parse_args()

    file_types = [ext.strip() for ext in args.file_types.split(',')]

    batch_process_models(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        normalize=not args.no_normalize,
        max_workers=args.workers,
        file_types=file_types
    )


if __name__ == "__main__":
    main()