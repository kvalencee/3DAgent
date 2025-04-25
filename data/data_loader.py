import os
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Voxelizer:
    """Clase para convertir mallas 3D en representaciones voxelizadas."""

    def __init__(self, resolution=32):
        """
        Inicializa el voxelizador.

        Args:
            resolution (int): Resolución de la representación voxelizada (e.g., 32 para un grid de 32x32x32)
        """
        self.resolution = resolution

    def voxelize(self, mesh_path):
        """
        Convierte una malla 3D en una representación voxelizada.

        Args:
            mesh_path (str): Ruta al archivo de malla (.obj, .stl, etc.)

        Returns:
            np.ndarray: Representación voxelizada de la malla (1 donde hay material, 0 donde no)
        """
        try:
            # Cargar la malla
            mesh = trimesh.load(mesh_path)

            # Normalizar la malla para que quepa en un cubo unitario
            mesh.vertices -= mesh.bounding_box.centroid
            max_dim = max(mesh.bounding_box.extents)
            mesh.vertices /= max_dim

            # Voxelizar la malla
            voxels = mesh.voxelized(self.resolution)

            # Convertir a matriz binaria 3D
            voxel_grid = voxels.matrix

            # Asegurarse de que la forma es correcta (resolution x resolution x resolution)
            if voxel_grid.shape != (self.resolution, self.resolution, self.resolution):
                voxel_grid_resized = np.zeros((self.resolution, self.resolution, self.resolution), dtype=bool)
                min_dims = [min(voxel_grid.shape[i], self.resolution) for i in range(3)]
                voxel_grid_resized[:min_dims[0], :min_dims[1], :min_dims[2]] = voxel_grid[:min_dims[0], :min_dims[1],
                                                                               :min_dims[2]]
                voxel_grid = voxel_grid_resized

            return voxel_grid.astype(np.float32)
        except Exception as e:
            print(f"Error al voxelizar {mesh_path}: {e}")
            return np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)


class Decorative3DDataset(Dataset):
    """Dataset para modelos 3D decorativos."""

    def __init__(self, root_dir, transform=None, resolution=32, category_filter=None):
        """
        Inicializa el dataset.

        Args:
            root_dir (str): Directorio raíz que contiene los modelos 3D
            transform (callable, optional): Transformaciones opcionales
            resolution (int): Resolución de la representación voxelizada
            category_filter (list, optional): Lista de categorías a incluir
        """
        self.root_dir = root_dir
        self.transform = transform
        self.resolution = resolution
        self.voxelizer = Voxelizer(resolution=resolution)

        # Buscar todos los archivos de modelo 3D
        self.model_paths = []
        self.categories = []
        self.labels = []

        # Extensiones de archivo a considerar
        valid_extensions = ['.obj', '.stl', '.ply']

        for root, _, files in os.walk(root_dir):
            # Obtener la categoría basada en el nombre de la carpeta
            rel_path = os.path.relpath(root, root_dir)
            category = rel_path.split(os.sep)[0] if rel_path != '.' else 'uncategorized'

            if category_filter and category not in category_filter:
                continue

            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    file_path = os.path.join(root, file)
                    self.model_paths.append(file_path)
                    self.categories.append(category)

                    # Si necesitamos un ID numérico para cada categoría
                    if category not in self.labels:
                        self.labels.append(category)

        print(f"Encontrados {len(self.model_paths)} modelos 3D en {len(set(self.categories))} categorías.")

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, idx):
        model_path = self.model_paths[idx]
        category = self.categories[idx]
        label_idx = self.labels.index(category)

        # Voxelizar el modelo
        voxel_data = self.voxelizer.voxelize(model_path)

        # Añadir dimensión de canal (1 x resolution x resolution x resolution)
        voxel_tensor = torch.tensor(voxel_data).float().unsqueeze(0)

        # Aplicar transformaciones si existen
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)

        return {
            'voxels': voxel_tensor,
            'category': category,
            'label_idx': label_idx,
            'path': model_path
        }


def create_dataloader(root_dir, batch_size=16, resolution=32, category_filter=None):
    """
    Crea un dataloader para el dataset de modelos 3D.

    Args:
        root_dir (str): Directorio raíz con los modelos
        batch_size (int): Tamaño del batch
        resolution (int): Resolución de voxelización
        category_filter (list, optional): Filtro de categorías

    Returns:
        DataLoader: Dataloader para el dataset
    """
    dataset = Decorative3DDataset(
        root_dir=root_dir,
        resolution=resolution,
        category_filter=category_filter
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return dataloader, dataset


# Ejemplo de uso:
if __name__ == "__main__":
    # Este código se ejecutará si ejecutas este archivo directamente
    data_dir = "sample_data/"  # Reemplaza con la ruta a tus datos

    # Crear un dataloader
    dataloader, dataset = create_dataloader(data_dir, batch_size=4, resolution=32)

    # Mostrar información sobre el primer batch
    for batch in dataloader:
        print(f"Batch shape: {batch['voxels'].shape}")
        print(f"Categories: {batch['category']}")
        break