# printability_checker.py
import os
import argparse
import json
import trimesh
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_printability(model_path):
    """
    Analiza la imprimibilidad de un modelo 3D

    Args:
        model_path: Ruta al modelo

    Returns:
        dict: Resultados del análisis
    """
    try:
        # Cargar modelo
        mesh = trimesh.load(model_path)

        # Resultados
        results = {
            'filename': os.path.basename(model_path),
            'filepath': model_path,
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'checks': {}
        }

        # 1. Verificar si es estanco (watertight)
        results['checks']['is_watertight'] = {
            'result': mesh.is_watertight,
            'score': 1.0 if mesh.is_watertight else 0.0,
            'description': 'El modelo es estanco' if mesh.is_watertight else 'El modelo no es estanco (tiene huecos)'
        }

        # 2. Verificar si es una variedad 2D (manifold)
        results['checks']['is_manifold'] = {
            'result': mesh.is_manifold,
            'score': 1.0 if mesh.is_manifold else 0.0,
            'description': 'El modelo es una variedad 2D' if mesh.is_manifold else 'El modelo no es una variedad 2D'
        }

        # 3. Verificar si hay caras degeneradas
        degenerate_faces = np.sum(mesh.area_faces < 1e-8)
        degenerate_percentage = degenerate_faces / len(mesh.faces) if len(mesh.faces) > 0 else 0
        degenerate_score = max(0, 1.0 - degenerate_percentage * 10)  # Penalizar si > 10% caras degeneradas

        results['checks']['degenerate_faces'] = {
            'result': degenerate_faces == 0,
            'score': degenerate_score,
            'value': int(degenerate_faces),
            'percentage': float(degenerate_percentage * 100),
            'description': f'Caras degeneradas: {degenerate_faces} ({degenerate_percentage:.2%})'
        }

        # 4. Verificar orientación de normales
        normals_consistent = True

        if hasattr(mesh, 'face_normals'):
            # Verificar si hay normales inconsistentes
            if len(mesh.face_normals) > 1:
                # Calcular ángulos entre normales adyacentes
                face_adjacency = mesh.face_adjacency
                if len(face_adjacency) > 0:
                    face_normals_angle = np.abs(np.arccos(np.clip(
                        np.sum(mesh.face_normals[face_adjacency[:, 0]] *
                               mesh.face_normals[face_adjacency[:, 1]], axis=1),
                        -1.0, 1.0
                    )))

                    # Contar ángulos mayores a 90 grados
                    inconsistent_count = np.sum(face_normals_angle > np.pi / 2)
                    inconsistent_percentage = inconsistent_count / len(face_adjacency)

                    normals_consistent = inconsistent_percentage < 0.1  # < 10% inconsistencias
                    normal_score = max(0, 1.0 - inconsistent_percentage * 2)  # Penalizar si > 50% inconsistencias
                else:
                    normal_score = 0.5  # No hay adyacencia para verificar
            else:
                normal_score = 1.0  # Solo una cara
        else:
            normal_score = 0.5  # No hay normales

        results['checks']['normals_consistent'] = {
            'result': normals_consistent,
            'score': normal_score,
            'description': 'Normales consistentes' if normals_consistent else 'Normales inconsistentes'
        }

        # 5. Verificar proporciones y estabilidad
        dimensions = mesh.bounding_box.extents
        height = max(dimensions)
        base_area = dimensions[0] * dimensions[1]  # Aproximación

        aspect_ratio = height / min(dimensions[0], dimensions[1]) if min(dimensions[0], dimensions[1]) > 0 else float(
            'inf')

        stability_score = 1.0
        if aspect_ratio > 10:
            stability_score
    stability_score = 1.0
    if aspect_ratio > 10:
        stability_score = 0.0  # Muy inestable
    elif aspect_ratio > 5:
        stability_score = 0.5  # Moderadamente inestable

    results['checks']['stability'] = {
        'result': stability_score > 0.7,
        'score': stability_score,
        'value': float(aspect_ratio),
        'description': f'Relación altura/base: {aspect_ratio:.2f}'
    }

    # 6. Verificar volumen mínimo
    has_volume = False
    volume_score = 0.0

    if mesh.is_watertight:
        volume = mesh.volume
        has_volume = volume > 0

        # Volumen mínimo para impresión (muy dependiente de la tecnología)
        if volume > 1.0:  # cm³
            volume_score = 1.0
        else:
            volume_score = min(1.0, volume)

    results['checks']['has_volume'] = {
        'result': has_volume,
        'score': volume_score,
        'value': float(mesh.volume) if mesh.is_watertight else 0.0,
        'description': f'Volumen: {mesh.volume:.2f} cm³' if mesh.is_watertight else 'No se puede calcular volumen'
    }

    # 7. Verificar superficie mínima
    area = mesh.area
    area_score = min(1.0, area / 10.0)  # Score máximo a partir de 10 cm²

    results['checks']['surface_area'] = {
        'result': area > 1.0,  # Mínimo 1 cm²
        'score': area_score,
        'value': float(area),
        'description': f'Área de superficie: {area:.2f} cm²'
    }

    # Calcular puntuación global de imprimibilidad
    scores = [check['score'] for check in results['checks'].values()]
    results['printability_score'] = float(np.mean(scores))

    # Categoría de imprimibilidad (1-4)
    if results['printability_score'] > 0.9:
        results['printability_category'] = 1  # Fácil de imprimir
    elif results['printability_score'] > 0.7:
        results['printability_category'] = 2  # Impresión estándar
    elif results['printability_score'] > 0.5:
        results['printability_category'] = 3  # Impresión desafiante
    else:
        results['printability_category'] = 4  # Difícil de imprimir

    # Necesidad de soportes
    results['needs_supports'] = aspect_ratio > 4 or not mesh.is_watertight

    return results

    except Exception as e:
    return {
        'filename': os.path.basename(model_path),
        'filepath': model_path,
        'error': str(e),
        'printability_score': 0.0,
        'printability_category': 4,
        'needs_supports': True
    }


def batch_analyze_printability(input_dir, output_file=None, visualize=True):
    """
    Analiza la imprimibilidad de todos los modelos 3D en un directorio

    Args:
        input_dir: Directorio con modelos 3D
        output_file: Archivo de salida para resultados
        visualize: Si se debe generar visualizaciones

    Returns:
        list: Resultados del análisis
    """
    # Buscar archivos
    model_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.stl', '.obj', '.ply')):
                model_files.append(os.path.join(root, file))

    print(f"Analizando {len(model_files)} modelos 3D...")

    # Analizar cada modelo
    results = []
    for model_path in tqdm(model_files, desc="Analizando imprimibilidad"):
        result = analyze_printability(model_path)
        results.append(result)

    # Crear DataFrame para análisis
    df = pd.DataFrame(results)

    # Guardar resultados
    if output_file:
        if output_file.lower().endswith('.csv'):
            df.to_csv(output_file, index=False)
        elif output_file.lower().endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        else:
            # Por defecto, guardar como CSV
            df.to_csv(output_file, index=False)

    # Generar visualizaciones
    if visualize:
        # Directorio para visualizaciones
        vis_dir = os.path.join(os.path.dirname(output_file) if output_file else input_dir, "printability_analysis")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. Distribución de puntuaciones de imprimibilidad
        plt.figure(figsize=(10, 6))
        sns.histplot(df['printability_score'].dropna(), bins=20, kde=True)
        plt.title('Distribución de Puntuaciones de Imprimibilidad')
        plt.xlabel('Puntuación (0-1)')
        plt.ylabel('Número de Modelos')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(vis_dir, "printability_scores.png"), dpi=300)
        plt.close()

        # 2. Distribución por categoría
        plt.figure(figsize=(10, 6))
        cat_counts = df['printability_category'].value_counts().sort_index()
        sns.barplot(x=cat_counts.index, y=cat_counts.values)
        plt.title('Distribución por Categoría de Imprimibilidad')
        plt.xlabel('Categoría (1-4)')
        plt.ylabel('Número de Modelos')
        plt.xticks([0, 1, 2, 3], ['1 - Fácil', '2 - Estándar', '3 - Desafiante', '4 - Difícil'])
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(os.path.join(vis_dir, "printability_categories.png"), dpi=300)
        plt.close()

        # 3. Necesidad de soportes
        plt.figure(figsize=(8, 6))
        support_counts = df['needs_supports'].value_counts()
        plt.pie(support_counts.values, labels=['No', 'Sí'] if False in support_counts.index else ['Sí'],
                autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Necesidad de Soportes')
        plt.savefig(os.path.join(vis_dir, "needs_supports.png"), dpi=300)
        plt.close()

        # 4. Resumen de verificaciones
        checks_summary = {}
        for result in results:
            if 'checks' in result:
                for check_name, check_data in result['checks'].items():
                    if check_name not in checks_summary:
                        checks_summary[check_name] = []
                    checks_summary[check_name].append(check_data.get('result', False))

        checks_df = pd.DataFrame({
            'Verificación': list(checks_summary.keys()),
            'Tasa de Éxito': [sum(results) / len(results) * 100 for results in checks_summary.values()]
        })

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Verificación', y='Tasa de Éxito', data=checks_df)
        plt.title('Tasa de Éxito por Verificación')
        plt.xlabel('Verificación')
        plt.ylabel('Tasa de Éxito (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "checks_success_rate.png"), dpi=300)
        plt.close()

        print(f"Visualizaciones guardadas en {vis_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analiza la imprimibilidad de modelos 3D")
    parser.add_argument('--input_dir', required=True, help='Directorio con modelos 3D')
    parser.add_argument('--output_file', help='Archivo de salida para resultados')
    parser.add_argument('--no_visualize', action='store_true', help='No generar visualizaciones')

    args = parser.parse_args()

    # Configurar archivo de salida por defecto
    if not args.output_file:
        args.output_file = os.path.join(args.input_dir, "printability_results.csv")

    # Ejecutar análisis
    batch_analyze_printability(
        input_dir=args.input_dir,
        output_file=args.output_file,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()