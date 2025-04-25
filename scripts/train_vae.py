import os
import argparse
import time
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Añadir la carpeta raíz al path de forma adecuada
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ahora importar desde las rutas relativas
from models.vae_model import VoxelVAE, vae_loss
from data.data_loader import Decorative3DDataset, create_dataloader


def train_vae(args):
    """
    Entrena un VAE para modelos 3D.

    Args:
        args: Argumentos de línea de comandos
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Usando dispositivo: {device}")

    # Crear dataloader
    train_dataloader, train_dataset = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        resolution=args.resolution,
        category_filter=args.categories.split(',') if args.categories else None
    )

    # Crear modelo
    model = VoxelVAE(resolution=args.resolution, latent_dim=args.latent_dim).to(device)

    # Crear optimizador
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Crear scheduler para ajustar learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Historial de pérdidas
    train_losses = []
    bce_losses = []
    kld_losses = []

    # Crear directorio para checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.output_dir, f"vae_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Entrenar modelo
    print(f"Iniciando entrenamiento por {args.epochs} épocas...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_bce = 0
        epoch_kld = 0

        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch + 1}/{args.epochs}")

        for batch in progress_bar:
            # Obtener datos y moverlos al dispositivo
            voxels = batch['voxels'].to(device)

            # Reset del gradiente
            optimizer.zero_grad()

            # Forward pass
            recon_voxels, mu, log_var = model(voxels)

            # Calcular pérdida
            loss, bce, kld = vae_loss(recon_voxels, voxels, mu, log_var, beta=args.beta)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            # Actualizar pérdidas
            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_kld += kld.item()

            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'loss': loss.item() / voxels.size(0),
                'bce': bce.item() / voxels.size(0),
                'kld': kld.item() / voxels.size(0)
            })

        # Actualizar scheduler
        scheduler.step()

        # Calcular pérdidas promedio
        avg_loss = epoch_loss / len(train_dataset)
        avg_bce = epoch_bce / len(train_dataset)
        avg_kld = epoch_kld / len(train_dataset)

        # Guardar historial
        train_losses.append(avg_loss)
        bce_losses.append(avg_bce)
        kld_losses.append(avg_kld)

        # Imprimir estadísticas
        print(f"Época {epoch + 1}/{args.epochs} - Pérdida: {avg_loss:.4f} - BCE: {avg_bce:.4f} - KLD: {avg_kld:.4f}")

        # Guardar checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'bce': avg_bce,
                'kld': avg_kld
            }, checkpoint_path)
            print(f"Checkpoint guardado en {checkpoint_path}")

            # Generar y guardar algunas muestras
            if args.save_samples:
                save_samples(model, device, checkpoint_dir, epoch + 1, args.resolution)

    # Calcular tiempo total
    total_time = time.time() - start_time
    print(f"Entrenamiento completado en {total_time:.2f} segundos")

    # Guardar modelo final
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modelo final guardado en {final_model_path}")

    # Guardar gráfica de pérdidas
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Total Loss')
    plt.plot(bce_losses, label='BCE Loss')
    plt.plot(kld_losses, label='KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, "training_loss.png"))
    plt.close()

    return model, checkpoint_dir


def save_samples(model, device, output_dir, epoch, resolution, num_samples=5):
    """
    Genera y guarda muestras del modelo.

    Args:
        model: Modelo entrenado
        device: Dispositivo (CPU/GPU)
        output_dir: Directorio de salida
        epoch: Época actual
        resolution: Resolución del modelo
        num_samples: Número de muestras a generar
    """
    # Cambiar a modo evaluación
    model.eval()

    # Crear directorio para muestras
    samples_dir = os.path.join(output_dir, f"samples_epoch_{epoch}")
    os.makedirs(samples_dir, exist_ok=True)

    # Generar muestras
    with torch.no_grad():
        # Muestras aleatorias
        samples = model.sample(num_samples=num_samples, device=device)

        # Guardar cada muestra como una visualización 3D
        for i, sample in enumerate(samples):
            # Convertir a numpy
            voxels = sample.squeeze().cpu().numpy()

            # Crear visualización
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Convertir voxels a coordenadas x, y, z donde voxel > 0.5
            x, y, z = np.indices((resolution, resolution, resolution))
            voxels_binary = voxels > 0.5
            ax.voxels(voxels_binary, edgecolor='k', alpha=0.2)

            # Configurar visualización
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Muestra {i + 1}')

            # Guardar figura
            plt.savefig(os.path.join(samples_dir, f"sample_{i + 1}.png"))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de VAE para modelos 3D")

    # Directorios
    parser.add_argument('--data_dir', type=str, required=True, help='Directorio de datos')
    parser.add_argument('--output_dir', type=str, default='output', help='Directorio de salida')

    # Hiperparámetros del modelo
    parser.add_argument('--resolution', type=int, default=32, help='Resolución del modelo')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimensión del espacio latente')
    parser.add_argument('--beta', type=float, default=1.0, help='Peso del término KL en la pérdida')

    # Hiperparámetros de entrenamiento
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Tasa de aprendizaje')
    parser.add_argument('--lr_step_size', type=int, default=30, help='Frecuencia de ajuste de lr')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Factor de ajuste de lr')

    # Otros
    parser.add_argument('--save_freq', type=int, default=10, help='Frecuencia de guardado de checkpoints')
    parser.add_argument('--save_samples', action='store_true', help='Guardar muestras generadas')
    parser.add_argument('--no_cuda', action='store_true', help='Desactivar CUDA')
    parser.add_argument('--categories', type=str, default=None, help='Categorías a usar (separadas por coma)')

    args = parser.parse_args()

    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)

    # Entrenar modelo
    model, checkpoint_dir = train_vae(args)

    print(f"Entrenamiento completado. Resultados guardados en {checkpoint_dir}")


if __name__ == "__main__":
    main()