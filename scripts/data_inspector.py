# data_inspector.py
import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from tqdm import tqdm


def inspect_voxel_dataset(dataset_dir, output_dir=None, sample_count=5):
    """
    Inspecciona un conjunto de datos de voxels y genera visualizaciones

    Args:
        dataset_dir: Directorio con archivos numpy de voxels
        output_dir: Directorio para guardar visualizaciones y estadísticas
        sample_count: Número de muestras a visualizar

    Returns:
        dict: Estadísticas del conjunto de datos
    """
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, "inspection")

    os.makedirs(output_dir, exist_ok=True)

    # Encontrar todos los archivos numpy
    voxel_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".npy"):
                voxel_files.append(os.path.join(root, file))

    if not voxel_files:
        print(f"No se encontraron archivos de voxels en {dataset_dir}")
        return {}

    print(f"Analizando {len(voxel_files)} archivos de voxels...")

    # Estadísticas a recopilar
    stats = {
        "total_files": len(voxel_files),
        "resolutions": {},
        "occupancy_stats": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        },
        "categories": {}
    }

    # Analizar archivos
    occupancies = []

    for file_path in tqdm(voxel_files, desc="Analizando voxels"):
        try:
            # Obtener categoría (basado en directorio padre)
            category = os.path.basename(os.path.dirname(file_path))
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1

            # Cargar voxels
            voxels = np.load(file_path)

            # Si tiene dimensión de batch o canal, extraer el primer elemento
            if voxels.ndim > 3:
                voxels = voxels[0]

            # Si todavía tiene dimensión de canal, extraer
            if voxels.ndim > 3:
                voxels = voxels[0]

            # Registrar resolución
            resolution = voxels.shape[0]
            resolution_key = f"{resolution}x{resolution}x{resolution}"
            if resolution_key not in stats["resolutions"]:
                stats["resolutions"][resolution_key] = 0
            stats["resolutions"][resolution_key] += 1

            # Calcular ocupación (porcentaje de voxels ocupados)
            occupancy = np.mean(voxels > 0.5)
            occupancies.append(occupancy)

            # Actualizar estadísticas de ocupación
            stats["occupancy_stats"]["min"] = min(stats["occupancy_stats"]["min"], occupancy)
            stats["occupancy_stats"]["max"] = max(stats["occupancy_stats"]["max"], occupancy)

        except Exception as e:
            print(f"Error analizando {file_path}: {e}")

    # Calcular ocupación media
    if occupancies:
        stats["occupancy_stats"]["avg"] = np.mean(occupancies)
    else:
        stats["occupancy_stats"]["min"] = 0

    # Visualizar algunas muestras
    if voxel_files and sample_count > 0:
        sample_files = np.random.choice(voxel_files, min(sample_count, len(voxel_files)), replace=False)

        for i, file_path in enumerate(sample_files):
            try:
                # Cargar voxels
                voxels = np.load(file_path)

                # Si tiene dimensión de batch o canal, extraer el primer elemento
                if voxels.ndim > 3:
                    voxels = voxels[0]

                # Si todavía tiene dimensión de canal, extraer
                if voxels.ndim > 3:
                    voxels = voxels[0]

                # Crear visualización
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

                # Visualizar voxels
                voxels_binary = voxels > 0.5
                ax.voxels(voxels_binary, edgecolor='k', alpha=0.3)

                # Ajustar visualización
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # Título con info del archivo
                file_name = os.path.basename(file_path)
                category = os.path.basename(os.path.dirname(file_path))
                occupancy = np.mean(voxels_binary)
                ax.set_title(f"{category}/{file_name}\nOcupación: {occupancy:.2%}")

                # Guardar visualización
                plt.savefig(os.path.join(output_dir, f"sample_{i + 1}.png"))
                plt.close(fig)

            except Exception as e:
                print(f"Error visualizando {file_path}: {e}")

    # Visualizar distribución de ocupación
    if occupancies:
        plt.figure(figsize=(10, 6))
        plt.hist(occupancies, bins=20)
        plt.xlabel('Ocupación (porcentaje de voxels ocupados)')
        plt.ylabel('Número de modelos')
        plt.title('Distribución de ocupación en el conjunto de datos')
        plt.savefig(os.path.join(output_dir, "occupancy_distribution.png"))
        plt.close()

    # Visualizar distribución por categoría
    if stats["categories"]:
        categories = list(stats["categories"].keys())
        counts = list(stats["categories"].values())

        plt.figure(figsize=(12, 6))
        plt.bar(categories, counts)
        plt.xlabel('Categoría')
        plt.ylabel('Número de modelos')
        plt.title('Distribución por categoría')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "category_distribution.png"))
        plt.close()

    # Guardar estadísticas
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Inspección completada. Resultados guardados en {output_dir}")
    print(f"Total de archivos: {stats['total_files']}")
    print(f"Resoluciones encontradas: {stats['resolutions']}")
    print(f"Ocupación promedio: {stats['occupancy_stats']['avg']:.2%}")

    return stats


def inspect_mesh_dataset(dataset_dir, output_dir=None, sample_count=5):
    """
    Inspecciona un conjunto de datos de mallas y genera visualizaciones

    Args:
        dataset_dir: Directorio con archivos de mallas (STL, OBJ, PLY)
        output_dir: Directorio para guardar visualizaciones y estadísticas
        sample_count: Número de muestras a visualizar

    Returns:
        dict: Estadísticas del conjunto de datos
    """
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, "inspection")

    os.makedirs(output_dir, exist_ok=True)

    # Encontrar todos los archivos de malla
    mesh_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".stl", ".obj", ".ply", ".off")):
                mesh_files.append(os.path.join(root, file))

    if not mesh_files:
        print(f"No se encontraron archivos de malla en {dataset_dir}")
        return {}

    print(f"Analizando {len(mesh_files)} archivos de malla...")

    # Estadísticas a recopilar
    stats = {
        "total_files": len(mesh_files),
        "file_types": {},
        "vertex_stats": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        },
        "face_stats": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        },
        "categories": {}
    }

    # Analizar archivos
    vertex_counts = []
    face_counts = []

    for file_path in tqdm(mesh_files, desc="Analizando mallas"):
        try:
            # Obtener categoría (basado en directorio padre)
            category = os.path.basename(os.path.dirname(file_path))
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1

            # Registrar tipo de archivo
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in stats["file_types"]:
                stats["file_types"][file_ext] = 0
            stats["file_types"][file_ext] += 1
            # Cargar malla
            mesh = trimesh.load(file_path)

            # Contar vértices y caras
            vertex_count = len(mesh.vertices)
            face_count = len(mesh.faces)

            vertex_counts.append(vertex_count)
            face_counts.append(face_count)

            # Actualizar estadísticas
            stats["vertex_stats"]["min"] = min(stats["vertex_stats"]["min"], vertex_count)
            stats["vertex_stats"]["max"] = max(stats["vertex_stats"]["max"], vertex_count)

            stats["face_stats"]["min"] = min(stats["face_stats"]["min"], face_count)
            stats["face_stats"]["max"] = max(stats["face_stats"]["max"], face_count)

        except Exception as e:
            print(f"Error analizando {file_path}: {e}")

        # Calcular promedios
        if vertex_counts:
            stats["vertex_stats"]["avg"] = np.mean(vertex_counts)
        else:
            stats["vertex_stats"]["min"] = 0

        if face_counts:
            stats["face_stats"]["avg"] = np.mean(face_counts)
        else:
            stats["face_stats"]["min"] = 0

        # Visualizar algunas muestras
        if mesh_files and sample_count > 0:
            sample_files = np.random.choice(mesh_files, min(sample_count, len(mesh_files)), replace=False)

            for i, file_path in enumerate(sample_files):
                try:
                    # Cargar malla
                    mesh = trimesh.load(file_path)

                    # Renderizar vista
                    scene = trimesh.Scene(mesh)

                    # Crear visualización desde diferentes ángulos
                    fig = plt.figure(figsize=(15, 5))

                    # Vista frontal
                    ax1 = fig.add_subplot(131, projection='3d')
                    scene.camera_transform = trimesh.transformations.rotation_matrix(
                        np.pi / 2, [0, 1, 0], scene.centroid)
                    png = scene.save_image(resolution=[300, 300])
                    from PIL import Image
                    import io
                    ax1.imshow(np.array(Image.open(io.BytesIO(png))))
                    ax1.set_title("Vista frontal")
                    ax1.axis('off')

                    # Vista superior
                    ax2 = fig.add_subplot(132, projection='3d')
                    scene.camera_transform = trimesh.transformations.rotation_matrix(
                        np.pi / 2, [1, 0, 0], scene.centroid)
                    png = scene.save_image(resolution=[300, 300])
                    ax2.imshow(np.array(Image.open(io.BytesIO(png))))
                    ax2.set_title("Vista superior")
                    ax2.axis('off')

                    # Vista isométrica
                    ax3 = fig.add_subplot(133, projection='3d')
                    scene.camera_transform = trimesh.transformations.rotation_matrix(
                        np.pi / 4, [1, 0, 0], scene.centroid)
                    scene.camera_transform = trimesh.transformations.rotation_matrix(
                        np.pi / 4, [0, 0, 1], scene.centroid).dot(scene.camera_transform)
                    png = scene.save_image(resolution=[300, 300])
                    ax3.imshow(np.array(Image.open(io.BytesIO(png))))
                    ax3.set_title("Vista isométrica")
                    ax3.axis('off')

                    # Título con info del archivo
                    file_name = os.path.basename(file_path)
                    category = os.path.basename(os.path.dirname(file_path))
                    plt.suptitle(f"{category}/{file_name}\nVértices: {len(mesh.vertices)}, Caras: {len(mesh.faces)}")

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"mesh_sample_{i + 1}.png"))
                    plt.close(fig)

                except Exception as e:
                    print(f"Error visualizando {file_path}: {e}")

        # Visualizar distribución de vértices
        if vertex_counts:
            plt.figure(figsize=(10, 6))
            plt.hist(vertex_counts, bins=20)
            plt.xlabel('Número de vértices')
            plt.ylabel('Número de modelos')
            plt.title('Distribución de complejidad (vértices) en el conjunto de datos')
            plt.savefig(os.path.join(output_dir, "vertex_distribution.png"))
            plt.close()

        # Visualizar distribución de caras
        if face_counts:
            plt.figure(figsize=(10, 6))
            plt.hist(face_counts, bins=20)
            plt.xlabel('Número de caras')
            plt.ylabel('Número de modelos')
            plt.title('Distribución de complejidad (caras) en el conjunto de datos')
            plt.savefig(os.path.join(output_dir, "face_distribution.png"))
            plt.close()

        # Visualizar distribución por categoría
        if stats["categories"]:
            categories = list(stats["categories"].keys())
            counts = list(stats["categories"].values())

            plt.figure(figsize=(12, 6))
            plt.bar(categories, counts)
            plt.xlabel('Categoría')
            plt.ylabel('Número de modelos')
            plt.title('Distribución por categoría')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "mesh_category_distribution.png"))
            plt.close()

        # Guardar estadísticas
        stats_path = os.path.join(output_dir, "mesh_dataset_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Inspección completada. Resultados guardados en {output_dir}")
        print(f"Total de archivos: {stats['total_files']}")
        print(f"Tipos de archivo: {stats['file_types']}")
        print(f"Vértices promedio: {stats['vertex_stats']['avg']:.2f}")
        print(f"Caras promedio: {stats['face_stats']['avg']:.2f}")

        return stats

    def main():
        parser = argparse.ArgumentParser(description="Inspeccionar conjuntos de datos 3D")

        parser.add_argument('--data_dir', type=str, required=True,
                            help='Directorio con datos a inspeccionar')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Directorio para guardar resultados')
        parser.add_argument('--type', type=str, choices=['voxel', 'mesh'], required=True,
                            help='Tipo de datos a inspeccionar')
        parser.add_argument('--sample_count', type=int, default=5,
                            help='Número de muestras a visualizar')

        args = parser.parse_args()

        if args.type == 'voxel':
            inspect_voxel_dataset(
                dataset_dir=args.data_dir,
                output_dir=args.output_dir,
                sample_count=args.sample_count
            )
        elif args.type == 'mesh':
            inspect_mesh_dataset(
                dataset_dir=args.data_dir,
                output_dir=args.output_dir,
                sample_count=args.sample_count
            )

    if __name__ == "__main__":
        main()