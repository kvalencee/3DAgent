import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from skimage import measure
import os
import torch
import mcubes  # Para marching cubes


def visualize_voxel_grid(voxels, threshold=0.5, fig=None, ax=None, title=None, alpha=0.5, save_path=None):
    """
    Visualiza una representación voxelizada como una malla 3D.

    Args:
        voxels (np.ndarray): Representación voxelizada (valores de 0 a 1)
        threshold (float): Umbral para considerar un voxel como sólido
        fig (matplotlib.figure): Figura existente o None para crear una nueva
        ax (matplotlib.axes): Ejes existentes o None para crear nuevos
        title (str): Título para la visualización
        alpha (float): Transparencia de la visualización
        save_path (str): Ruta para guardar la visualización o None

    Returns:
        fig, ax: Figura y ejes de matplotlib
    """
    # Asegurarse de que voxels sea numpy array
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()

    # Si voxels tiene 4 dimensiones (batch, channels, height, width, depth), tomar el primer elemento
    if voxels.ndim == 5:
        voxels = voxels[0]  # Tomar primer elemento del batch

    # Si voxels tiene canal, eliminar dimensión de canal
    if voxels.ndim == 4:
        voxels = voxels[0]  # Eliminar dimensión de canal

    # Crear figura y ejes si no se proporcionan
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Binarizar voxels según el umbral
    voxels_binary = voxels > threshold

    # Obtener posiciones de los voxels
    x, y, z = np.indices(voxels.shape)

    # Visualizar voxels
    ax.voxels(voxels_binary, edgecolor='k', alpha=alpha)

    # Configurar visualización
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if title:
        ax.set_title(title)

    # Guardar visualización si se proporciona ruta
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig, ax


def voxel_to_mesh(voxels, threshold=0.5, method='marching_cubes'):
    """
    Convierte una representación voxelizada en una malla 3D usando marching cubes.

    Args:
        voxels (np.ndarray): Representación voxelizada (valores de 0 a 1)
        threshold (float): Umbral para considerar un voxel como sólido
        method (str): Método de conversión ('marching_cubes' o 'marching_cubes_lewiner')

    Returns:
        trimesh.Trimesh: Malla 3D
    """
    # Asegurarse de que voxels sea numpy array
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()

    # Si voxels tiene 4 dimensiones (batch, channels, height, width, depth), tomar el primer elemento
    if voxels.ndim == 5:
        voxels = voxels[0]  # Tomar primer elemento del batch

    # Si voxels tiene canal, eliminar dimensión de canal
    if voxels.ndim == 4:
        voxels = voxels[0]  # Eliminar dimensión de canal

    # Aplicar marching cubes para obtener vértices y caras
    if method == 'marching_cubes':
        # Usar mcubes para una mejor implementación de marching cubes
        vertices, triangles = mcubes.marching_cubes(voxels, threshold)

        # Normalizar vértices al rango [-0.5, 0.5]
        vertices = vertices / voxels.shape[0] - 0.5

    elif method == 'marching_cubes_lewiner':
        # Usar skimage para marching cubes (alternativa)
        vertices, faces, normals, _ = measure.marching_cubes(voxels, threshold)

        # Normalizar vértices al rango [-0.5, 0.5]
        vertices = vertices / voxels.shape[0] - 0.5

        # Renombrar para consistencia
        triangles = faces
    else:
        raise ValueError(f"Método desconocido: {method}")

    # Crear malla con trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh


def export_mesh(mesh, output_path, format='stl'):
    """
    Exporta una malla 3D a un archivo.

    Args:
        mesh (trimesh.Trimesh): Malla 3D
        output_path (str): Ruta de salida sin extensión
        format (str): Formato de salida ('stl', 'obj', 'ply', etc.)

    Returns:
        str: Ruta completa del archivo guardado
    """
    # Asegurarse de que la carpeta de salida existe
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Añadir extensión si no está presente
    if not output_path.lower().endswith(f'.{format}'):
        output_path = f"{output_path}.{format}"

    # Exportar malla
    mesh.export(output_path)

    return output_path


def convert_voxels_to_mesh_file(voxels, output_path, threshold=0.5, format='stl', method='marching_cubes'):
    """
    Convierte directamente voxels a un archivo de malla 3D.

    Args:
        voxels (np.ndarray o torch.Tensor): Representación voxelizada
        output_path (str): Ruta de salida sin extensión
        threshold (float): Umbral para considerar un voxel como sólido
        format (str): Formato de salida ('stl', 'obj', 'ply', etc.)
        method (str): Método de conversión ('marching_cubes' o 'marching_cubes_lewiner')

    Returns:
        str: Ruta completa del archivo guardado
    """
    # Convertir voxels a malla
    mesh = voxel_to_mesh(voxels, threshold, method)

    # Exportar malla
    return export_mesh(mesh, output_path, format)


def visualize_batch(batch_voxels, threshold=0.5, rows=None, cols=None, figsize=(20, 20), save_path=None):
    """
    Visualiza un batch de voxels en una cuadrícula.

    Args:
        batch_voxels (torch.Tensor o np.ndarray): Batch de voxels (B, C, D, H, W)
        threshold (float): Umbral para considerar un voxel como sólido
        rows (int): Número de filas en la cuadrícula
        cols (int): Número de columnas en la cuadrícula
        figsize (tuple): Tamaño de la figura
        save_path (str): Ruta para guardar la visualización o None

    Returns:
        matplotlib.figure.Figure: Figura de matplotlib
    """
    # Asegurarse de que batch_voxels sea numpy array
    if isinstance(batch_voxels, torch.Tensor):
        batch_voxels = batch_voxels.detach().cpu().numpy()

    # Obtener número de elementos en el batch
    batch_size = batch_voxels.shape[0]

    # Determinar filas y columnas de la cuadrícula
    if rows is None and cols is None:
        # Calcular cuadrícula cuadrada aproximada
        cols = int(np.ceil(np.sqrt(batch_size)))
        rows = int(np.ceil(batch_size / cols))
    elif rows is None:
        rows = int(np.ceil(batch_size / cols))
    elif cols is None:
        cols = int(np.ceil(batch_size / rows))

    # Crear figura
    fig = plt.figure(figsize=figsize)

    # Iterar por cada elemento del batch
    for i in range(min(batch_size, rows * cols)):
        # Obtener subgráfico
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # Visualizar voxels
        visualize_voxel_grid(batch_voxels[i], threshold=threshold, fig=fig, ax=ax,
                             title=f"Sample {i + 1}", alpha=0.5)

    # Ajustar espaciado
    plt.tight_layout()

    # Guardar visualización si se proporciona ruta
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo
    resolution = 32
    voxels = np.random.rand(1, 1, resolution, resolution, resolution)

    # Visualizar
    fig, ax = visualize_voxel_grid(voxels, title="Voxel Grid Example")
    plt.show()

    # Convertir a malla y exportar
    output_path = "example_mesh"
    saved_path = convert_voxels_to_mesh_file(voxels, output_path, format='stl')
    print(f"Mesh saved to: {saved_path}")