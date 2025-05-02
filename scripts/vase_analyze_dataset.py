import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import trimesh
import argparse
from tqdm import tqdm


def analyze_vase_dataset(models_dir, metadata_file, output_dir):
    """
    Analiza un conjunto de datos de floreros 3D y genera estadísticas e informes.

    Args:
        models_dir: Directorio con modelos 3D
        metadata_file: Archivo CSV con etiquetas
        output_dir: Directorio para guardar resultados
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Cargar metadatos
    if not os.path.exists(metadata_file):
        print(f"Error: No se encontró el archivo de metadatos {metadata_file}")
        return

    df = pd.read_csv(metadata_file)
    print(f"Analizando {len(df)} modelos de floreros...")

    # Análisis técnico de los modelos
    if os.path.exists(models_dir):
        stats = analyze_models_technical(models_dir, df, output_dir)
        df = pd.merge(df, stats, on='filename', how='left')

    # Generar visualizaciones
    generate_visualizations(df, output_dir)

    # Generar informe de estadísticas
    generate_statistics_report(df, output_dir)

    # Guardar DataFrame actualizado
    df.to_csv(os.path.join(output_dir, "vase_dataset_analyzed.csv"), index=False)

    print(f"Análisis completado. Resultados guardados en {output_dir}")
    return df


def analyze_models_technical(models_dir, metadata_df, output_dir):
    """
    Analiza técnicamente los modelos 3D y genera estadísticas.

    Args:
        models_dir: Directorio con modelos 3D
        metadata_df: DataFrame con metadatos
        output_dir: Directorio para guardar resultados

    Returns:
        DataFrame con estadísticas técnicas
    """
    stats = []

    print("Analizando propiedades técnicas de los modelos...")
    for filename in tqdm(metadata_df['filename']):
        # Buscar archivo en el directorio de modelos
        model_path = None
        for root, _, files in os.walk(models_dir):
            if filename in files:
                model_path = os.path.join(root, filename)
                break

        if not model_path:
            # Intentar con extensiones comunes
            for ext in ['.stl', '.obj', '.ply']:
                base_name = os.path.splitext(filename)[0]
                for root, _, files in os.walk(models_dir):
                    possible_file = f"{base_name}{ext}"
                    if possible_file in files:
                        model_path = os.path.join(root, possible_file)
                        break
                if model_path:
                    break

        if not model_path:
            stats.append({
                'filename': filename,
                'vertices': np.nan,
                'faces': np.nan,
                'is_watertight': np.nan,
                'volume': np.nan,
                'surface_area': np.nan,
                'dimensions_x': np.nan,
                'dimensions_y': np.nan,
                'dimensions_z': np.nan
            })
            continue

        try:
            mesh = trimesh.load(model_path)

            # Calcular estadísticas
            dimensions = mesh.bounding_box.extents

            stats.append({
                'filename': filename,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'is_watertight': mesh.is_watertight,
                'volume': mesh.volume if mesh.is_watertight else np.nan,
                'surface_area': mesh.area,
                'dimensions_x': dimensions[0],
                'dimensions_y': dimensions[1],
                'dimensions_z': dimensions[2]
            })
        except Exception as e:
            print(f"Error analizando {filename}: {str(e)}")
            stats.append({
                'filename': filename,
                'vertices': np.nan,
                'faces': np.nan,
                'is_watertight': np.nan,
                'volume': np.nan,
                'surface_area': np.nan,
                'dimensions_x': np.nan,
                'dimensions_y': np.nan,
                'dimensions_z': np.nan
            })

    # Convertir a DataFrame
    stats_df = pd.DataFrame(stats)

    # Guardar estadísticas
    stats_df.to_csv(os.path.join(output_dir, "technical_stats.csv"), index=False)

    return stats_df


def generate_visualizations(df, output_dir):
    """
    Genera visualizaciones a partir del DataFrame de metadatos.

    Args:
        df: DataFrame con metadatos
        output_dir: Directorio para guardar visualizaciones
    """
    # Crear directorio para visualizaciones
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Configurar estilo
    plt.style.use('ggplot')
    sns.set(style="whitegrid")

    # 1. Distribución de tipos de florero
    plt.figure(figsize=(10, 6))
    type_counts = df['type'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values)
    plt.title('Distribución de Tipos de Florero')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "vase_types_distribution.png"), dpi=300)
    plt.close()

    # 2. Distribución de estilos
    plt.figure(figsize=(10, 6))
    style_counts = df['style'].value_counts()
    sns.barplot(x=style_counts.index, y=style_counts.values)
    plt.title('Distribución de Estilos de Florero')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "vase_styles_distribution.png"), dpi=300)
    plt.close()

    # 3. Distribución de complejidad
    plt.figure(figsize=(10, 6))
    # Extraer solo el número de complejidad (1, 2, 3, 4) si está en formato "1 - Simple"
    df['complexity_num'] = df['complexity'].str.extract(r'(\d+)', expand=False).astype(float)
    sns.countplot(x='complexity_num', data=df)
    plt.title('Distribución de Niveles de Complejidad')
    plt.xlabel('Nivel de Complejidad')
    plt.ylabel('Número de Modelos')
    plt.savefig(os.path.join(vis_dir, "complexity_distribution.png"), dpi=300)
    plt.close()

    # 4. Relación entre complejidad y número de vértices
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='complexity_num', y='vertices', data=df)
    plt.title('Relación entre Complejidad y Número de Vértices')
    plt.xlabel('Nivel de Complejidad')
    plt.ylabel('Número de Vértices')
    plt.savefig(os.path.join(vis_dir, "complexity_vs_vertices.png"), dpi=300)
    plt.close()

    # 5. Correlación entre propiedades numéricas
    numeric_cols = ['vertices', 'faces', 'volume', 'surface_area',
                    'dimensions_x', 'dimensions_y', 'dimensions_z']
    numeric_df = df[numeric_cols].copy()

    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')
    plt.title('Correlación entre Propiedades Numéricas')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "property_correlation.png"), dpi=300)
    plt.close()

    # 6. Análisis por tipo de florero
    # Mostrar propiedades promedio por tipo de florero
    type_stats = df.groupby('type').agg({
        'vertices': 'mean',
        'faces': 'mean',
        'volume': 'mean',
        'surface_area': 'mean',
        'dimensions_z': 'mean'  # Altura
    }).reset_index()

    # Graficar vertices por tipo
    plt.figure(figsize=(12, 6))
    sns.barplot(x='type', y='vertices', data=type_stats)
    plt.title('Promedio de Vértices por Tipo de Florero')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "vertices_by_type.png"), dpi=300)
    plt.close()

    # Graficar altura por tipo
    plt.figure(figsize=(12, 6))
    sns.barplot(x='type', y='dimensions_z', data=type_stats)
    plt.title('Altura Promedio por Tipo de Florero')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "height_by_type.png"), dpi=300)
    plt.close()

    # 7. Análisis de características de impresión
    if 'needs_supports' in df.columns:
        plt.figure(figsize=(8, 6))
        support_counts = df['needs_supports'].value_counts()
        plt.pie(support_counts.values, labels=support_counts.index, autopct='%1.1f%%',
                startangle=90, shadow=True)
        plt.title('Proporción de Modelos que Requieren Soportes')
        plt.axis('equal')
        plt.savefig(os.path.join(vis_dir, "support_requirements.png"), dpi=300)
        plt.close()

    # 8. Distribución de dimensiones
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.histplot(df['dimensions_x'], kde=True, ax=axes[0])
    axes[0].set_title('Distribución de Ancho (X)')

    sns.histplot(df['dimensions_y'], kde=True, ax=axes[1])
    axes[1].set_title('Distribución de Profundidad (Y)')

    sns.histplot(df['dimensions_z'], kde=True, ax=axes[2])
    axes[2].set_title('Distribución de Altura (Z)')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "dimensions_distribution.png"), dpi=300)
    plt.close()

    print(f"Visualizaciones guardadas en {vis_dir}")


def generate_statistics_report(df, output_dir):
    """
    Genera un informe de estadísticas del conjunto de datos.

    Args:
        df: DataFrame con metadatos
        output_dir: Directorio para guardar informe
    """
    # Crear informe
    report_path = os.path.join(output_dir, "dataset_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Informe de Análisis de Dataset de Floreros 3D\n\n")

        # Información general
        f.write("## Información General\n\n")
        f.write(f"* **Número total de modelos:** {len(df)}\n")
        f.write(f"* **Número de modelos imprimibles (water-tight):** {df['is_watertight'].sum()}\n")
        f.write(f"* **Fecha de análisis:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        # Estadísticas de propiedades
        f.write("## Estadísticas de Propiedades Geométricas\n\n")

        # Tabla de estadísticas numéricas
        num_stats = df[['vertices', 'faces', 'volume', 'surface_area',
                        'dimensions_x', 'dimensions_y', 'dimensions_z']].describe()

        f.write("### Resumen Estadístico\n\n")
        f.write(
            "| Estadística | Vértices | Caras | Volumen | Área de Superficie | Ancho (X) | Profundidad (Y) | Altura (Z) |\n")
        f.write(
            "|------------|----------|-------|---------|-------------------|----------|----------------|----------|\n")

        for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            row = f"| **{stat}** | "
            for col in ['vertices', 'faces', 'volume', 'surface_area',
                        'dimensions_x', 'dimensions_y', 'dimensions_z']:
                val = num_stats.loc[stat, col]
                if pd.isna(val):
                    row += "N/A | "
                elif col in ['vertices', 'faces']:
                    row += f"{val:.0f} | "
                else:
                    row += f"{val:.2f} | "
            f.write(row + "\n")

        f.write("\n")

        # Distribución por tipo
        f.write("## Distribución por Categorías\n\n")

        f.write("### Tipos de Florero\n\n")
        type_counts = df['type'].value_counts()
        f.write("| Tipo | Cantidad | Porcentaje |\n")
        f.write("|------|----------|------------|\n")
        for type_name, count in type_counts.items():
            if pd.isna(type_name):
                continue
            percentage = count / len(df) * 100
            f.write(f"| {type_name} | {count} | {percentage:.1f}% |\n")

        f.write("\n")

        f.write("### Estilos de Florero\n\n")
        style_counts = df['style'].value_counts()
        f.write("| Estilo | Cantidad | Porcentaje |\n")
        f.write("|--------|----------|------------|\n")
        for style_name, count in style_counts.items():
            if pd.isna(style_name):
                continue
            percentage = count / len(df) * 100
            f.write(f"| {style_name} | {count} | {percentage:.1f}% |\n")

        f.write("\n")

        f.write("### Niveles de Complejidad\n\n")
        complexity_counts = df['complexity'].value_counts()
        f.write("| Complejidad | Cantidad | Porcentaje |\n")
        f.write("|------------|----------|------------|\n")
        for complexity, count in complexity_counts.items():
            if pd.isna(complexity):
                continue
            percentage = count / len(df) * 100
            f.write(f"| {complexity} | {count} | {percentage:.1f}% |\n")

        f.write("\n")

        # Características de impresión
        if 'needs_supports' in df.columns and 'print_time_estimate' in df.columns:
            f.write("## Características de Impresión 3D\n\n")

            # Necesidad de soportes
            support_counts = df['needs_supports'].value_counts()
            if not support_counts.empty:
                f.write("### Necesidad de Soportes\n\n")
                f.write("| Necesita Soportes | Cantidad | Porcentaje |\n")
                f.write("|-------------------|----------|------------|\n")
                for support, count in support_counts.items():
                    if pd.isna(support):
                        continue
                    percentage = count / len(df) * 100
                    f.write(f"| {support} | {count} | {percentage:.1f}% |\n")

                f.write("\n")

            # Tiempo de impresión y uso de filamento
            f.write("### Estadísticas de Impresión\n\n")

            # Convertir a numérico si es posible
            if 'print_time_estimate' in df.columns:
                df['print_time_numeric'] = pd.to_numeric(df['print_time_estimate'], errors='coerce')

            if 'filament_usage' in df.columns:
                df['filament_numeric'] = pd.to_numeric(df['filament_usage'], errors='coerce')

            if 'print_time_numeric' in df.columns and 'filament_numeric' in df.columns:
                print_stats = df[['print_time_numeric', 'filament_numeric']].describe()

                f.write("| Estadística | Tiempo de Impresión (h) | Uso de Filamento (g) |\n")
                f.write("|-------------|--------------------------|----------------------|\n")

                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    time_val = print_stats.loc[
                        stat, 'print_time_numeric'] if 'print_time_numeric' in print_stats else np.nan
                    filament_val = print_stats.loc[
                        stat, 'filament_numeric'] if 'filament_numeric' in print_stats else np.nan

                    time_str = f"{time_val:.2f}" if not pd.isna(time_val) else "N/A"
                    filament_str = f"{filament_val:.2f}" if not pd.isna(filament_val) else "N/A"

                    f.write(f"| **{stat}** | {time_str} | {filament_str} |\n")

                f.write("\n")

        # Conclusiones
        f.write("## Conclusiones y Recomendaciones\n\n")

        # Calcular propiedades medias por tipo
        type_stats = df.groupby('type').agg({
            'vertices': 'mean',
            'faces': 'mean',
            'volume': 'mean',
            'dimensions_z': 'mean'
        }).reset_index()

        # Encontrar el tipo más complejo (mayor número de vértices)
        most_complex_type = type_stats.loc[type_stats['vertices'].idxmax(), 'type']

        # Encontrar el tipo más alto
        tallest_type = type_stats.loc[type_stats['dimensions_z'].idxmax(), 'type']

        f.write(f"* El tipo de florero más complejo (mayor número de vértices) es: **{most_complex_type}**\n")
        f.write(f"* El tipo de florero más alto en promedio es: **{tallest_type}**\n")

        # Calcular correlación entre complejidad y vértices
        if 'complexity_num' in df.columns and 'vertices' in df.columns:
            complexity_corr = df['complexity_num'].corr(df['vertices'])
            f.write(
                f"* La correlación entre complejidad percibida y número de vértices es: **{complexity_corr:.2f}**\n")

            if complexity_corr > 0.7:
                f.write(
                    "  * Existe una fuerte correlación positiva entre la complejidad percibida y el número de vértices.\n")
            elif complexity_corr > 0.4:
                f.write("  * Existe una correlación moderada entre la complejidad percibida y el número de vértices.\n")
            else:
                f.write(
                    "  * No hay una fuerte correlación entre la complejidad percibida y el número de vértices, lo que sugiere que la complejidad percibida puede depender de otros factores como la geometría o el detalle artístico.\n")

        # Recomendaciones para impresión 3D
        watertight_percentage = df['is_watertight'].sum() / len(df) * 100 if 'is_watertight' in df.columns else np.nan

        f.write("\n### Recomendaciones para Impresión 3D\n\n")
        if not pd.isna(watertight_percentage):
            f.write(
                f"* **{watertight_percentage:.1f}%** de los modelos son adecuados para impresión 3D sin reparación (water-tight).\n")

            if watertight_percentage < 70:
                f.write("* Se recomienda revisar y reparar la mayoría de los modelos antes de imprimir.\n")
            else:
                f.write("* La mayoría de los modelos están listos para imprimir directamente.\n")

        # Más recomendaciones
        f.write(
            "* Considere simplificar modelos con más de 100,000 vértices para mejorar el rendimiento de impresión.\n")
        f.write(
            "* Para modelos complejos, considere aumentar el grosor de pared para garantizar resistencia estructural.\n")

    print(f"Informe guardado en {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analizar conjunto de datos de floreros 3D")
    parser.add_argument('--models_dir', required=True, help='Directorio con modelos 3D')
    parser.add_argument('--metadata_file', required=True, help='Archivo CSV con etiquetas')
    parser.add_argument('--output_dir', default='analysis_results', help='Directorio para guardar resultados')

    args = parser.parse_args()
    analyze_vase_dataset(args.models_dir, args.metadata_file, args.output_dir)