# batch_process.py
import os
import sys
import argparse
import subprocess
import trimesh
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

def normalize_model(input_path, output_path=None, scale=True, center=True, repair=True):
    """
    Normaliza un modelo 3D:
    - Escala a tamaño unitario
    - Centra en el origen
    - Repara problemas comunes

    Args:
        input_path: Ruta al modelo de entrada
        output_path: Ruta de salida (opcional)
        scale: Si se debe escalar
        center: Si se debe centrar
        repair: Si se deben reparar problemas

    Returns:
        trimesh.Trimesh: Modelo normalizado
        str: Ruta de salida
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
            # Asegurarse de que las normales apuntan hacia afuera
            mesh.fix_normals()

        # Guardar modelo normalizado
        mesh.export(output_path)

        return mesh, output_path

    except Exception as e:
        print(f"Error procesando {input_path}: {str(e)}")
        return None, None

def analyze_model(mesh, model_path):
    """
    Analiza un modelo 3D y extrae propiedades importantes

    Args:
        mesh: Modelo 3D (trimesh.Trimesh)
        model_path: Ruta al modelo

    Returns:
        dict: Propiedades del modelo
    """
    filename = os.path.basename(model_path)

    try:
        # Propiedades básicas
        props = {
            'filename': filename,
            'filepath': model_path,
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'is_watertight': mesh.is_watertight,
            'is_manifold': getattr(mesh, 'is_manifold', False),
            'volume': None,
            'surface_area': mesh.area,
            'dimensions': mesh.bounding_box.extents.tolist(),
            'height': float(max(mesh.bounding_box.extents)),
            'width': float(min(mesh.bounding_box.extents)),
            'aspect_ratio': float(max(mesh.bounding_box.extents) / min(mesh.bounding_box.extents))
                            if min(mesh.bounding_box.extents) > 0 else float('inf')
        }

        # Calcular volumen si es estanco
        if mesh.is_watertight:
            props['volume'] = float(mesh.volume)

        # Evaluar imprimibilidad
        printability = 1  # 1-4, donde 1 es más fácil de imprimir
        if not mesh.is_watertight:
            printability += 2
        if props['aspect_ratio'] > 5:
            printability += 1

        props['printability'] = min(printability, 4)

        # Estimar si necesita soportes
        needs_supports = False
        if props['aspect_ratio'] > 4 or not mesh.is_watertight:
            needs_supports = True

        props['needs_supports'] = needs_supports

        # Estimar tiempo de impresión y filamento (muy aproximado)
        if props.get('volume'):
            # Fórmula básica: 1 cm³ ≈ 10 minutos (depende de muchos factores)
            props['print_time_estimate'] = props['volume'] / 6  # Horas
            # Filamento: 1 cm³ ≈ 1.25g (PLA)
            props['filament_usage'] = props['volume'] * 1.25

        return props

    except Exception as e:
        print(f"Error analizando {model_path}: {str(e)}")
        return {
            'filename': filename,
            'filepath': model_path,
            'error': str(e)
        }

def process_model(input_path, output_dir, analyze_only=False):
    """
    Procesa un modelo 3D: normaliza y analiza

    Args:
        input_path: Ruta al modelo
        output_dir: Directorio de salida
        analyze_only: Si solo se debe analizar sin normalizar

    Returns:
        dict: Propiedades del modelo
    """
    try:
        # Crear ruta de salida
        basename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, basename)

        if not analyze_only:
            # Normalizar modelo
            mesh, output_path = normalize_model(input_path, output_path)
            if mesh is None:
                return {
                    'filename': basename,
                    'filepath': input_path,
                    'error': 'Error al normalizar'
                }
        else:
            # Solo cargar modelo
            mesh = trimesh.load(input_path)

        # Analizar modelo
        props = analyze_model(mesh, output_path if not analyze_only else input_path)
        return props

    except Exception as e:
        return {
            'filename': os.path.basename(input_path),
            'filepath': input_path,
            'error': str(e)
        }

def batch_process_models(input_dir, output_dir=None, normalize=True, max_workers=4, file_types=None):
    """
    Procesa por lotes todos los modelos 3D en un directorio

    Args:
        input_dir: Directorio de entrada
        output_dir: Directorio de salida (opcional)
        normalize: Si se deben normalizar los modelos
        max_workers: Número máximo de procesos simultáneos
        file_types: Lista de extensiones de archivo a procesar

    Returns:
        dict: Resultados del procesamiento
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_model, model_path, output_dir, not normalize)
            for model_path in model_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando modelos"):
            result = future.result()
            results.append(result)

            if 'error' in result:
                error_count += 1
            else:
                processed_count += 1

    # Guardar resultados
    stats = {
        'total': len(model_files),
        'processed': processed_count,
        'errors': error_count,
        'details': results
    }

    stats_path = os.path.join(output_dir, "processing_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"Procesamiento completado: {processed_count} exitosos, {error_count} errores")
    print(f"Resultados guardados en: {stats_path}")

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