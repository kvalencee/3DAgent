import os
import sys
import argparse
import torch
import numpy as np
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Añadir la carpeta raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decorai.models.gan_model import WGAN_GP, ConditionalWGAN_GP
from decorai.utils.visualization import visualize_voxel_grid, convert_voxels_to_mesh_file, visualize_batch


def load_model(checkpoint_path, config_path=None, device='cuda'):
    """
    Carga un modelo GAN desde un checkpoint.

    Args:
        checkpoint_path: Ruta al checkpoint del modelo
        config_path: Ruta al archivo de configuración (opcional)
        device: Dispositivo donde cargar el modelo

    Returns:
        Modelo GAN cargado y configuración
    """
    # Cargar configuración si se proporciona
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"Configuración cargada desde {config_path}")
    else:
        # Intentar buscar la configuración en el mismo directorio que el checkpoint
        config_dir = os.path.dirname(checkpoint_path)
        potential_config_path = os.path.join(config_dir, 'config.json')

        if os.path.exists(potential_config_path):
            with open(potential_config_path, 'r') as f:
                config = json.load(f)

            print(f"Configuración cargada desde {potential_config_path}")

    # Si no se encuentra configuración, usar valores predeterminados
    if config is None:
        print("No se encontró configuración, usando valores predeterminados")
        config = {
            'latent_dim': 200,
            'channels': 64,
            'resolution': 32,
            'num_classes': 5,
            'model_type': 'wgan_gp'  # Asumir modelo no condicional
        }

    # Crear modelo
    if config.get('model_type') == 'conditional_wgan_gp' or config.get('conditional', False):
        model = ConditionalWGAN_GP(
            latent_dim=config.get('latent_dim', 200),
            channels=config.get('channels', 64),
            voxel_size=config.get('resolution', 32),
            num_classes=config.get('num_classes', 5)
        ).to(device)
        print("Cargando GAN condicional")
    else:
        model = WGAN_GP(
            latent_dim=config.get('latent_dim', 200),
            channels=config.get('channels', 64),
            voxel_size=config.get('resolution', 32)
        ).to(device)
        print("Cargando GAN no condicional")

    # Cargar pesos
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Diferentes formatos de checkpoint
        if 'generator_state_dict' in checkpoint:
            model.generator.load_state_dict(checkpoint['generator_state_dict'])
            model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            # Intentar cargar directamente
            model.load_state_dict(checkpoint)

        print(f"Modelo cargado desde {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint_path}")

    # Cambiar a modo evaluación
    model.eval()

    return model, config


def generate_models(args):
    """
    Genera modelos 3D con un GAN entrenado.

    Args:
        args: Argumentos de línea de comandos
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar modelo
    model, config = load_model(args.checkpoint_path, args.config_path, device)

    # Crear directorio de salida
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determinar si es condicional
    is_conditional = hasattr(model, 'generator') and hasattr(model.generator, 'class_embedding')

    # Determinar número de clases si es condicional
    num_classes = config.get('num_classes', 5) if is_conditional else 0

    # Generar modelos
    print(f"Generando {args.num_samples} modelos...")

    if is_conditional:
        # Para cada clase
        class_labels = range(num_classes) if args.all_classes else [args.class_id]

        for class_id in class_labels:
            # Crear subdirectorio para la clase
            class_dir = os.path.join(output_dir, f"class_{class_id}")
            os.makedirs(class_dir, exist_ok=True)

            # Generar modelos para esta clase
            print(f"Generando modelos para clase {class_id}...")

            for i in tqdm(range(args.num_samples)):
                # Generar una muestra
                labels = torch.full((1,), class_id, dtype=torch.long).to(device)
                sample, _ = model.generate_samples(1, labels, device)

                # Guardar visualización
                vis_path = os.path.join(class_dir, f"sample_{i + 1}.png")
                fig, ax = visualize_voxel_grid(
                    sample[0],
                    threshold=args.threshold,
                    title=f"Class {class_id} - Sample {i + 1}",
                    save_path=vis_path
                )
                plt.close(fig)

                # Guardar modelo 3D
                if args.export_mesh:
                    mesh_path = os.path.join(class_dir, f"sample_{i + 1}")
                    convert_voxels_to_mesh_file(
                        sample[0],
                        mesh_path,
                        threshold=args.threshold,
                        format=args.format
                    )

            # Generar una visualización de conjunto
            if args.visualize_grid:
                # Generar un conjunto de muestras para visualización
                num_grid_samples = min(16, args.num_samples)  # Máximo 16 para la cuadrícula
                grid_labels = torch.full((num_grid_samples,), class_id, dtype=torch.long).to(device)
                grid_samples, _ = model.generate_samples(num_grid_samples, grid_labels, device)

                # Guardar visualización de conjunto
                grid_path = os.path.join(class_dir, f"grid_samples.png")
                visualize_batch(
                    grid_samples,
                    threshold=args.threshold,
                    rows=int(np.sqrt(num_grid_samples)),
                    cols=int(np.sqrt(num_grid_samples)),
                    save_path=grid_path
                )
    else:
        # Generar modelos sin condicionamiento
        for i in tqdm(range(args.num_samples)):
            # Generar una muestra
            sample = model.generate_samples(1, device)

            # Guardar visualización
            vis_path = os.path.join(output_dir, f"sample_{i + 1}.png")
            fig, ax = visualize_voxel_grid(
                sample[0],
                threshold=args.threshold,
                title=f"Sample {i + 1}",
                save_path=vis_path
            )
            plt.close(fig)

            # Guardar modelo 3D
            if args.export_mesh:
                mesh_path = os.path.join(output_dir, f"sample_{i + 1}")
                convert_voxels_to_mesh_file(
                    sample[0],
                    mesh_path,
                    threshold=args.threshold,
                    format=args.format
                )

        # Generar una visualización de conjunto
        if args.visualize_grid:
            # Generar un conjunto de muestras para visualización
            num_grid_samples = min(16, args.num_samples)  # Máximo 16 para la cuadrícula
            grid_samples = model.generate_samples(num_grid_samples, device)

            # Guardar visualización de conjunto
            grid_path = os.path.join(output_dir, "grid_samples.png")
            visualize_batch(
                grid_samples,
                threshold=args.threshold,
                rows=int(np.sqrt(num_grid_samples)),
                cols=int(np.sqrt(num_grid_samples)),
                save_path=grid_path
            )

    print(f"Generación completada. Modelos guardados en {output_dir}")


def generate_interpolations(args):
    """
    Genera interpolaciones entre modelos 3D con un GAN entrenado.

    Args:
        args: Argumentos de línea de comandos
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar modelo
    model, config = load_model(args.checkpoint_path, args.config_path, device)

    # Crear directorio de salida
    output_dir = args.output_dir
    interp_dir = os.path.join(output_dir, "interpolations")
    os.makedirs(interp_dir, exist_ok=True)

    # Determinar si es condicional
    is_conditional = hasattr(model, 'generator') and hasattr(model.generator, 'class_embedding')

    # Determinar número de clases si es condicional
    num_classes = config.get('num_classes', 5) if is_conditional else 0

    # Obtener dimensión latente
    latent_dim = config.get('latent_dim', 200)

    # Generar puntos en el espacio latente
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)

    # Generar interpolaciones
    print(f"Generando {args.steps} interpolaciones...")

    if is_conditional:
        # Seleccionar clases para interpolación
        if args.interp_classes:
            # Usar clases especificadas
            class_ids = [int(c) for c in args.interp_classes.split(',')]
            if len(class_ids) != 2:
                print("Se necesitan exactamente 2 clases para interpolación. Usando 0 y 1.")
                class_ids = [0, 1]
        else:
            # Usar clases aleatorias
            class_ids = random.sample(range(num_classes), 2)

        # Crear etiquetas
        label1 = torch.tensor([class_ids[0]], device=device)
        label2 = torch.tensor([class_ids[1]], device=device)

        print(f"Interpolando entre clase {class_ids[0]} y clase {class_ids[1]}...")

        # Almacenar todas las muestras para visualización de cuadrícula
        all_samples = []

        # Interpolar entre los dos puntos
        for i in range(args.steps):
            # Calcular ratio de interpolación
            t = i / (args.steps - 1)

            # Interpolar latentes
            z_interp = z1 * (1 - t) + z2 * t

            # Interpolar etiquetas
            if args.interp_labels:
                # También interpolar las etiquetas
                interp_class = int(round(class_ids[0] * (1 - t) + class_ids[1] * t))
                label_interp = torch.tensor([interp_class], device=device)
            else:
                # Usar etiqueta dependiendo de qué lado estamos más cerca
                label_interp = label1 if t < 0.5 else label2

            # Generar muestra
            with torch.no_grad():
                sample = model.generator(z_interp, label_interp)

            # Almacenar para visualización
            all_samples.append(sample)

            # Guardar visualización individual
            vis_path = os.path.join(interp_dir, f"interp_{i + 1}.png")
            fig, ax = visualize_voxel_grid(
                sample[0],
                threshold=args.threshold,
                title=f"Interpolation {i + 1} (t={t:.2f}, class={label_interp.item()})",
                save_path=vis_path
            )
            plt.close(fig)

            # Guardar modelo 3D
            if args.export_mesh:
                mesh_path = os.path.join(interp_dir, f"interp_{i + 1}")
                convert_voxels_to_mesh_file(
                    sample[0],
                    mesh_path,
                    threshold=args.threshold,
                    format=args.format
                )

        # Concatenar muestras para visualización
        all_samples = torch.cat(all_samples, dim=0)

    else:
        # Almacenar todas las muestras para visualización de cuadrícula
        all_samples = []

        # Interpolar entre los dos puntos
        for i in range(args.steps):
            # Calcular ratio de interpolación
            t = i / (args.steps - 1)

            # Interpolar latentes
            z_interp = z1 * (1 - t) + z2 * t

            # Generar muestra
            with torch.no_grad():
                sample = model.generator(z_interp)

            # Almacenar para visualización
            all_samples.append(sample)

            # Guardar visualización individual
            vis_path = os.path.join(interp_dir, f"interp_{i + 1}.png")
            fig, ax = visualize_voxel_grid(
                sample[0],
                threshold=args.threshold,
                title=f"Interpolation {i + 1} (t={t:.2f})",
                save_path=vis_path
            )
            plt.close(fig)

            # Guardar modelo 3D
            if args.export_mesh:
                mesh_path = os.path.join(interp_dir, f"interp_{i + 1}")
                convert_voxels_to_mesh_file(
                    sample[0],
                    mesh_path,
                    threshold=args.threshold,
                    format=args.format
                )

        # Concatenar muestras para visualización
        all_samples = torch.cat(all_samples, dim=0)

    # Guardar visualización de todas las interpolaciones
    grid_path = os.path.join(interp_dir, "all_interpolations.png")
    fig = visualize_batch(
        all_samples,
        threshold=args.threshold,
        rows=1,
        cols=args.steps,
        figsize=(args.steps * 3, 3),
        save_path=grid_path
    )
    plt.close(fig)

    print(f"Interpolaciones completadas. Resultados guardados en {interp_dir}")


def explore_latent_space(args):
    """
    Explora el espacio latente del GAN.

    Args:
        args: Argumentos de línea de comandos
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar modelo
    model, config = load_model(args.checkpoint_path, args.config_path, device)

    # Crear directorio de salida
    output_dir = args.output_dir
    explore_dir = os.path.join(output_dir, "latent_exploration")
    os.makedirs(explore_dir, exist_ok=True)

    # Determinar si es condicional
    is_conditional = hasattr(model, 'generator') and hasattr(model.generator, 'class_embedding')

    # Dimensión latente
    latent_dim = config.get('latent_dim', 200)

    # Seleccionar dimensiones para exploración
    if args.dimensions:
        # Usar dimensiones especificadas
        dims = [int(d) for d in args.dimensions.split(',')]
        if len(dims) != 2:
            print("Se necesitan exactamente 2 dimensiones para exploración. Seleccionando aleatoriamente.")
            dims = random.sample(range(latent_dim), 2)
    else:
        # Seleccionar aleatoriamente
        dims = random.sample(range(latent_dim), 2)

    print(f"Explorando dimensiones latentes {dims[0]} y {dims[1]}...")

    # Crear grid
    grid_size = args.grid_size
    grid_values = np.linspace(-3.0, 3.0, grid_size)

    # Base latent vector (media cero)
    z_base = torch.zeros(1, latent_dim).to(device)

    if is_conditional:
        # Determinar número de clases
        num_classes = config.get('num_classes', 5)

        # Seleccionar clase
        class_id = args.class_id if args.class_id is not None else random.randint(0, num_classes - 1)
        label = torch.tensor([class_id], device=device)

        print(f"Explorando espacio latente para clase {class_id}...")

        # Almacenar todas las muestras para visualización
        all_samples = []

        # Explorar grid
        for i, val1 in enumerate(grid_values):
            row_samples = []

            for j, val2 in enumerate(grid_values):
                # Copiar vector base
                z = z_base.clone()

                # Modificar dimensiones seleccionadas
                z[0, dims[0]] = val1
                z[0, dims[1]] = val2

                # Generar muestra
                with torch.no_grad():
                    sample = model.generator(z, label)

                # Almacenar para visualización
                row_samples.append(sample)

                # Guardar visualización individual
                sample_name = f"dim{dims[0]}_{val1:.2f}_dim{dims[1]}_{val2:.2f}"
                vis_path = os.path.join(explore_dir, f"{sample_name}_class{class_id}.png")

                fig, ax = visualize_voxel_grid(
                    sample[0],
                    threshold=args.threshold,
                    title=f"Dim{dims[0]}={val1:.2f}, Dim{dims[1]}={val2:.2f}, Class={class_id}",
                    save_path=vis_path
                )
                plt.close(fig)

                # Guardar modelo 3D
                if args.export_mesh and i % 2 == 0 and j % 2 == 0:  # Guardar solo una selección para no sobrecargar
                    mesh_path = os.path.join(explore_dir, f"{sample_name}_class{class_id}")
                    convert_voxels_to_mesh_file(
                        sample[0],
                        mesh_path,
                        threshold=args.threshold,
                        format=args.format
                    )

            # Concatenar muestras de la fila
            all_samples.append(torch.cat(row_samples, dim=0))

        # Concatenar todas las filas
        all_samples = torch.cat(all_samples, dim=0)

    else:
        # Almacenar todas las muestras para visualización
        all_samples = []

        # Explorar grid
        for i, val1 in enumerate(grid_values):
            row_samples = []

            for j, val2 in enumerate(grid_values):
                # Copiar vector base
                z = z_base.clone()

                # Modificar dimensiones seleccionadas
                z[0, dims[0]] = val1
                z[0, dims[1]] = val2

                # Generar muestra
                with torch.no_grad():
                    sample = model.generator(z)

                # Almacenar para visualización
                row_samples.append(sample)

                # Guardar visualización individual
                sample_name = f"dim{dims[0]}_{val1:.2f}_dim{dims[1]}_{val2:.2f}"
                vis_path = os.path.join(explore_dir, f"{sample_name}.png")

                fig, ax = visualize_voxel_grid(
                    sample[0],
                    threshold=args.threshold,
                    title=f"Dim{dims[0]}={val1:.2f}, Dim{dims[1]}={val2:.2f}",
                    save_path=vis_path
                )
                plt.close(fig)

                # Guardar modelo 3D
                if args.export_mesh and i % 2 == 0 and j % 2 == 0:  # Guardar solo una selección para no sobrecargar
                    mesh_path = os.path.join(explore_dir, f"{sample_name}")
                    convert_voxels_to_mesh_file(
                        sample[0],
                        mesh_path,
                        threshold=args.threshold,
                        format=args.format
                    )

            # Concatenar muestras de la fila
            all_samples.append(torch.cat(row_samples, dim=0))

        # Concatenar todas las filas
        all_samples = torch.cat(all_samples, dim=0)

    # Guardar visualización de toda la cuadrícula
    grid_path = os.path.join(explore_dir, f"latent_grid_dims_{dims[0]}_{dims[1]}.png")
    fig = visualize_batch(
        all_samples,
        threshold=args.threshold,
        rows=grid_size,
        cols=grid_size,
        figsize=(grid_size * 2, grid_size * 2),
        save_path=grid_path
    )
    plt.close(fig)

    print(f"Exploración completada. Resultados guardados en {explore_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generar modelos 3D con GAN entrenado")

    # Parámetros generales
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Ruta al checkpoint del modelo')
    parser.add_argument('--config_path', type=str, default=None, help='Ruta al archivo de configuración')
    parser.add_argument('--output_dir', type=str, default='generated_models', help='Directorio de salida')
    parser.add_argument('--no_cuda', action='store_true', help='Desactivar CUDA')

    # Parámetros de generación
    parser.add_argument('--mode', type=str, choices=['generate', 'interpolate', 'explore'], default='generate',
                        help='Modo de generación')
    parser.add_argument('--num_samples', type=int, default=10, help='Número de muestras a generar')
    parser.add_argument('--class_id', type=int, default=0, help='ID de clase para generación condicional')
    parser.add_argument('--all_classes', action='store_true', help='Generar para todas las clases')
    parser.add_argument('--threshold', type=float, default=0.5, help='Umbral para binarización de voxels')

    # Parámetros para interpolación
    parser.add_argument('--steps', type=int, default=10, help='Número de pasos de interpolación')
    parser.add_argument('--interp_classes', type=str, default=None,
                        help='Clases para interpolación (separadas por coma)')
    parser.add_argument('--interp_labels', action='store_true', help='Interpolar entre etiquetas')

    # Parámetros para exploración
    parser.add_argument('--grid_size', type=int, default=5, help='Tamaño de la cuadrícula para exploración')
    parser.add_argument('--dimensions', type=str, default=None, help='Dimensiones a explorar (separadas por coma)')

    # Parámetros de visualización y exportación
    parser.add_argument('--visualize_grid', action='store_true', help='Visualizar cuadrícula de muestras')
    parser.add_argument('--export_mesh', action='store_true', help='Exportar mallas 3D')
    parser.add_argument('--format', type=str, default='stl', choices=['stl', 'obj', 'ply'],
                        help='Formato de exportación de mallas')

    args = parser.parse_args()

    # Ejecutar modo seleccionado
    if args.mode == 'generate':
        generate_models(args)
    elif args.mode == 'interpolate':
        generate_interpolations(args)
    elif args.mode == 'explore':
        explore_latent_space(args)


if __name__ == "__main__":
    main()