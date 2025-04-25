import os
import argparse
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json

# Añadir la carpeta raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decorai.models.gan_model import WGAN_GP, ConditionalWGAN_GP
from decorai.data.data_loader import create_dataloader
from decorai.utils.visualization import visualize_voxel_grid, convert_voxels_to_mesh_file, visualize_batch


def train_gan(args):
    """
    Entrena un modelo GAN para generación de modelos 3D.

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

    # Determinar número de clases
    num_classes = len(set(train_dataset.labels))
    print(f"Número de clases detectadas: {num_classes}")

    # Crear modelo GAN
    if args.conditional:
        model = ConditionalWGAN_GP(
            latent_dim=args.latent_dim,
            channels=args.channels,
            voxel_size=args.resolution,
            num_classes=num_classes
        ).to(device)
        print("Usando GAN condicional")
    else:
        model = WGAN_GP(
            latent_dim=args.latent_dim,
            channels=args.channels,
            voxel_size=args.resolution
        ).to(device)
        print("Usando GAN no condicional")

    # Crear optimizadores
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # Crear directorio para checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.output_dir, f"gan_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Configurar contadores y métricas
    num_train_steps = 0
    d_losses = []
    g_losses = []
    real_scores = []
    fake_scores = []

    # Número de pasos de generador por cada paso de discriminador
    n_critic = args.n_critic

    # Guardar configuración del experimento
    config = vars(args)
    config['num_classes'] = num_classes
    config['model_type'] = 'conditional_wgan_gp' if args.conditional else 'wgan_gp'

    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Entrenamiento
    print(f"Iniciando entrenamiento por {args.epochs} épocas...")

    for epoch in range(args.epochs):
        model.train()
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_real_score = 0
        epoch_fake_score = 0

        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch + 1}/{args.epochs}")

        for batch in progress_bar:
            batch_size = batch['voxels'].size(0)

            # Obtener datos y moverlos al dispositivo
            real_voxels = batch['voxels'].to(device)

            if args.conditional:
                labels = torch.tensor(batch['label_idx'], dtype=torch.long).to(device)

            # ----------------------
            # Entrenar Discriminador
            # ----------------------

            for _ in range(n_critic):
                optimizer_d.zero_grad()

                # Generar muestras falsas
                if args.conditional:
                    g_loss, fake_voxels, fake_labels = model.train_step_generator(batch_size, device, labels)
                    d_loss, real_score, fake_score = model.train_step_discriminator(
                        real_voxels, labels, fake_voxels, fake_labels, lambda_gp=args.lambda_gp, device=device
                    )
                else:
                    g_loss, fake_voxels = model.train_step_generator(batch_size, device)
                    d_loss, real_score, fake_score = model.train_step_discriminator(
                        real_voxels, fake_voxels, lambda_gp=args.lambda_gp, device=device
                    )

                # Backpropagation y optimización
                d_loss.backward()
                optimizer_d.step()

                # Actualizar métricas
                epoch_d_loss += d_loss.item()
                epoch_real_score += real_score.mean().item()
                epoch_fake_score += fake_score.mean().item()

            # ------------------
            # Entrenar Generador
            # ------------------

            optimizer_g.zero_grad()

            # Calcular pérdida del generador
            if args.conditional:
                g_loss, fake_voxels, _ = model.train_step_generator(batch_size, device, labels)
            else:
                g_loss, fake_voxels = model.train_step_generator(batch_size, device)

            # Backpropagation y optimización
            g_loss.backward()
            optimizer_g.step()

            # Actualizar métricas
            epoch_g_loss += g_loss.item()

            # Actualizar progreso
            num_train_steps += 1

            if num_train_steps % 10 == 0:
                # Actualizar barra de progreso cada 10 pasos
                progress_bar.set_postfix({
                    'D_loss': d_loss.item(),
                    'G_loss': g_loss.item(),
                    'Real': real_score.mean().item(),
                    'Fake': fake_score.mean().item()
                })

            # Guardar muestras periódicamente
            if num_train_steps % args.sample_interval == 0:
                with torch.no_grad():
                    if args.conditional:
                        # Generar muestras para cada clase
                        for class_idx in range(num_classes):
                            class_labels = torch.full((5,), class_idx, dtype=torch.long).to(device)
                            samples, _ = model.generate_samples(5, class_labels, device)

                            # Guardar visualización del batch
                            vis_path = os.path.join(
                                checkpoint_dir,
                                f"samples_step_{num_train_steps}_class_{class_idx}.png"
                            )
                            visualize_batch(samples, threshold=0.5, save_path=vis_path)

                            # Exportar un modelo 3D
                            if num_train_steps % (args.sample_interval * 5) == 0:
                                model_path = os.path.join(
                                    checkpoint_dir,
                                    f"model_step_{num_train_steps}_class_{class_idx}"
                                )
                                convert_voxels_to_mesh_file(samples[0], model_path)
                    else:
                        # Generar muestras aleatorias
                        samples = model.generate_samples(10, device)

                        # Guardar visualización del batch
                        vis_path = os.path.join(checkpoint_dir, f"samples_step_{num_train_steps}.png")
                        visualize_batch(samples, threshold=0.5, save_path=vis_path)

                        # Exportar un modelo 3D
                        if num_train_steps % (args.sample_interval * 5) == 0:
                            model_path = os.path.join(checkpoint_dir, f"model_step_{num_train_steps}")
                            convert_voxels_to_mesh_file(samples[0], model_path)

        # Calcular promedios de la época
        epoch_d_loss /= len(train_dataloader) * n_critic
        epoch_g_loss /= len(train_dataloader)
        epoch_real_score /= len(train_dataloader) * n_critic
        epoch_fake_score /= len(train_dataloader) * n_critic

        # Guardar métricas
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)
        real_scores.append(epoch_real_score)
        fake_scores.append(epoch_fake_score)

        # Imprimir estadísticas
        print(f"Época {epoch + 1}/{args.epochs} - D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, "
              f"Real Score: {epoch_real_score:.4f}, Fake Score: {epoch_fake_score:.4f}")

        # Guardar checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'd_loss': epoch_d_loss,
                'g_loss': epoch_g_loss
            }, checkpoint_path)
            print(f"Checkpoint guardado en {checkpoint_path}")

    # Guardar modelo final
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
    }, final_model_path)
    print(f"Modelo final guardado en {final_model_path}")

    # Guardar gráficas de pérdidas
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss')

    plt.subplot(1, 2, 2)
    plt.plot(real_scores, label='Real Scores')
    plt.plot(fake_scores, label='Fake Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Real and Fake Scores')

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "training_plots.png"))

    # Guardar métricas como CSV
    metrics = {
        'epoch': list(range(1, args.epochs + 1)),
        'd_loss': d_losses,
        'g_loss': g_losses,
        'real_score': real_scores,
        'fake_score': fake_scores
    }

    # Convertir a formato CSV
    metrics_csv = "epoch,d_loss,g_loss,real_score,fake_score\n"
    for i in range(args.epochs):
        metrics_csv += f"{metrics['epoch'][i]},{metrics['d_loss'][i]},{metrics['g_loss'][i]},"
        metrics_csv += f"{metrics['real_score'][i]},{metrics['fake_score'][i]}\n"

    with open(os.path.join(checkpoint_dir, "metrics.csv"), 'w') as f:
        f.write(metrics_csv)

    return model, checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de GAN para modelos 3D")

    # Directorios
    parser.add_argument('--data_dir', type=str, required=True, help='Directorio de datos')
    parser.add_argument('--output_dir', type=str, default='output', help='Directorio de salida')

    # Hiperparámetros del modelo
    parser.add_argument('--resolution', type=int, default=32, help='Resolución del modelo')
    parser.add_argument('--latent_dim', type=int, default=200, help='Dimensión del espacio latente')
    parser.add_argument('--channels', type=int, default=64, help='Número base de canales')
    parser.add_argument('--conditional', action='store_true', help='Usar GAN condicional')

    # Hiperparámetros de entrenamiento
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='Tasa de aprendizaje del generador')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='Tasa de aprendizaje del discriminador')
    parser.add_argument('--beta1', type=float, default=0.0, help='Beta1 para Adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='Beta2 para Adam')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Número de pasos del discriminador por cada paso del generador')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='Coeficiente de penalización de gradiente')

    # Otros
    parser.add_argument('--save_freq', type=int, default=10, help='Frecuencia de guardado de checkpoints')
    parser.add_argument('--sample_interval', type=int, default=100, help='Intervalo de pasos para generar muestras')
    parser.add_argument('--no_cuda', action='store_true', help='Desactivar CUDA')
    parser.add_argument('--categories', type=str, default=None, help='Categorías a usar (separadas por coma)')

    args = parser.parse_args()

    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)

    # Entrenar modelo
    model, checkpoint_dir = train_gan(args)

    print(f"Entrenamiento completado. Resultados guardados en {checkpoint_dir}")


if __name__ == "__main__":
    main()