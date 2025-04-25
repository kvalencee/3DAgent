import os
import argparse
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ahora importar desde las rutas relativas
from models.vae_model import VoxelVAE
from utils.visualization import visualize_voxel_grid, convert_voxels_to_mesh_file, visualize_batch

def load_model(model_path, resolution=32, latent_dim=128, device='cuda'):
    """
    Carga un modelo VAE previamente entrenado.

    Args:
        model_path: Ruta al archivo del modelo
        resolution: Resolución del modelo
        latent_dim: Dimensión del espacio latente
        device: Dispositivo (CPU/GPU)

    Returns:
        Modelo VAE cargado
    """
    # Crear modelo con la misma arquitectura
    model = VoxelVAE(resolution=resolution, latent_dim=latent_dim).to(device)

    # Cargar estado del modelo
    if os.path.isfile(model_path):
        # Diferentes formas de cargar dependiendo de cómo se guardó
        try:
            # Intentar cargar como state_dict directo
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            # Intentar cargar como checkpoint completo
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError(f"No se pudo cargar el modelo desde {model_path}")

        print(f"Modelo cargado de {model_path}")
    else:
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_path}")

    # Cambiar a modo evaluación
    model.eval()

    return model


def generate_random_samples(model, num_samples=5, device='cuda'):
    """
    Genera muestras aleatorias del modelo VAE.

    Args:
        model: Modelo VAE
        num_samples: Número de muestras a generar
        device: Dispositivo (CPU/GPU)

    Returns:
        Tensor con las muestras generadas
    """
    with torch.no_grad():
        # Generar muestras aleatorias
        samples = model.sample(num_samples=num_samples, device=device)

    return samples


def generate_interpolations(model, num_steps=10, device='cuda'):
    """
    Genera interpolaciones entre puntos aleatorios en el espacio latente.

    Args:
        model: Modelo VAE
        num_steps: Número de pasos de interpolación
        device: Dispositivo (CPU/GPU)

    Returns:
        Tensor con las interpolaciones generadas
    """
    # Generar dos puntos aleatorios en el espacio latente
    z1 = torch.randn(1, model.latent_dim).to(device)
    z2 = torch.randn(1, model.latent_dim).to(device)

    # Decodificar puntos para visualización
    with torch.no_grad():
        voxel1 = model.decode(z1)
        voxel2 = model.decode(z2)

        # Crear interpolaciones
        interpolations = []
        for alpha in np.linspace(0, 1, num_steps):
            # Interpolación lineal en el espacio latente
            z_interp = (1 - alpha) * z1 + alpha * z2
            # Decodificar
            voxel_interp = model.decode(z_interp)
            interpolations.append(voxel_interp)

        # Concatenar resultados (incluir originales)
        all_interpolations = torch.cat([voxel1] + interpolations + [voxel2], dim=0)

    return all_interpolations


def generate_latent_space_exploration(model, num_samples=25, device='cuda'):
    """
    Explora el espacio latente en una cuadrícula 2D.

    Args:
        model: Modelo VAE
        num_samples: Número de muestras (debe ser un cuadrado perfecto)
        device: Dispositivo (CPU/GPU)

    Returns:
        Tensor con las muestras generadas
    """
    # Verificar que num_samples es un cuadrado perfecto
    grid_size = int(np.sqrt(num_samples))
    if grid_size ** 2 != num_samples:
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        num_samples = grid_size ** 2
        print(f"Ajustando num_samples a {num_samples} para crear una cuadrícula {grid_size}x{grid_size}")

    # Crear cuadrícula 2D en espacio latente
    z_var = 1.0  # Varianza para muestreo

    # Seleccionar dos dimensiones aleatorias para explorar
    if model.latent_dim >= 2:
        dim1, dim2 = np.random.choice(model.latent_dim, size=2, replace=False)
    else:
        dim1, dim2 = 0, 0

    print(f"Explorando dimensiones latentes {dim1} y {dim2}")

    # Crear cuadrícula base de valores de z
    z_base = torch.zeros(num_samples, model.latent_dim).to(device)

    # Valores de cuadrícula
    linspace = np.linspace(-z_var, z_var, grid_size)

    # Llenar cuadrícula
    sample_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Asignar valores de cuadrícula a las dimensiones seleccionadas
            z_base[sample_idx, dim1] = linspace[i]
            z_base[sample_idx, dim2] = linspace[j]
            sample_idx += 1

    # Decodificar cuadrícula
    with torch.no_grad():
        samples = model.decode(z_base)

    return samples


def save_samples(samples, output_dir, prefix="sample", export_mesh=True, visualize=True, threshold=0.5):
    """
    Guarda las muestras generadas como imágenes y/o archivos de malla.

    Args:
        samples: Tensor con las muestras
        output_dir: Directorio de salida
        prefix: Prefijo para los archivos
        export_mesh: Si se deben exportar como mallas
        visualize: Si se deben visualizar
        threshold: Umbral para binarización
    """
    os.makedirs(output_dir, exist_ok=True)

    # Guardar cada muestra
    for i, sample in enumerate(samples):
        sample_path = os.path.join(output_dir, f"{prefix}_{i + 1}")

        # Visualizar
        if visualize:
            fig, ax = visualize_voxel_grid(
                sample,
                threshold=threshold,
                title=f"{prefix} {i + 1}",
                save_path=f"{sample_path}.png"
            )
            plt.close(fig)

        # Exportar como malla
        if export_mesh:
            mesh_path = convert_voxels_to_mesh_file(
                sample,
                sample_path,
                threshold=threshold,
                format='stl'
            )
            print(f"Malla guardada en {mesh_path}")


def main():
    parser = argparse.ArgumentParser(description="Generar modelos 3D con VAE")

    # Parámetros del modelo
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al modelo entrenado')
    parser.add_argument('--resolution', type=int, default=32, help='Resolución del modelo')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimensión del espacio latente')

    # Parámetros de generación
    parser.add_argument('--num_samples', type=int, default=5, help='Número de muestras a generar')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'interpolate', 'explore'],
                        help='Modo de generación')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Umbral para binarización de voxels')

    # Parámetros de salida
    parser.add_argument('--output_dir', type=str, default='generated_models',
                        help='Directorio de salida')
    parser.add_argument('--no_export_mesh', action='store_true',
                        help='No exportar como mallas')
    parser.add_argument('--no_visualize', action='store_true',
                        help='No visualizar muestras')
    parser.add_argument('--no_cuda', action='store_true',
                        help='No usar CUDA aunque esté disponible')

    args = parser.parse_args()

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar modelo
    model = load_model(
        args.model_path,
        resolution=args.resolution,
        latent_dim=args.latent_dim,
        device=device
    )

    # Generar muestras según el modo
    if args.mode == 'random':
        print(f"Generando {args.num_samples} muestras aleatorias...")
        samples = generate_random_samples(model, args.num_samples, device)
        output_subdir = os.path.join(args.output_dir, "random_samples")
        prefix = "random"

    elif args.mode == 'interpolate':
        print(f"Generando interpolación con {args.num_samples} pasos...")
        samples = generate_interpolations(model, args.num_samples, device)
        output_subdir = os.path.join(args.output_dir, "interpolations")
        prefix = "interp"

    elif args.mode == 'explore':
        print(f"Explorando espacio latente con {args.num_samples} muestras...")
        samples = generate_latent_space_exploration(model, args.num_samples, device)
        output_subdir = os.path.join(args.output_dir, "latent_space")
        prefix = "latent"

    # Guardar muestras
    save_samples(
        samples,
        output_subdir,
        prefix=prefix,
        export_mesh=not args.no_export_mesh,
        visualize=not args.no_visualize,
        threshold=args.threshold
    )

    # Visualizar todo el batch junto
    if not args.no_visualize:
        batch_vis_path = os.path.join(output_subdir, f"{prefix}_batch.png")
        fig = visualize_batch(samples, threshold=args.threshold, save_path=batch_vis_path)
        plt.close(fig)
        print(f"Visualización de batch guardada en {batch_vis_path}")

    print("¡Generación completada!")


if __name__ == "__main__":
    main()