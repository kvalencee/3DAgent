# train_enhanced_models.py
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
import wandb  # Para seguimiento y visualización

# Importar modelos mejorados
from models.enhanced_vae_model import EnhancedVoxelVAE, enhanced_vae_loss
from models.enhanced_gan_model import EnhancedWGAN_GP
from decorai.data.data_loader import create_dataloader


def train_vae(args):
    """
    Entrena el VAE mejorado con seguimiento de experimentos.
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Usando dispositivo: {device}")

    # Inicializar wandb para seguimiento
    if args.use_wandb:
        run = wandb.init(project="decorai-vae", name=f"vae-{args.resolution}-{datetime.now().strftime('%m%d-%H%M')}")
        wandb.config.update(args)

    # Crear dataloader
    train_dataloader, train_dataset = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        resolution=args.resolution,
        category_filter=args.categories.split(',') if args.categories else None
    )

    # Verificar si es condicional
    conditional = args.conditional
    num_classes = 0

    if conditional:
        # Contar clases únicas
        unique_classes = set()
        for batch in train_dataloader:
            if 'label_idx' in batch:
                unique_classes.update(batch['label_idx'])

        num_classes = len(unique_classes)
        print(f"Entrenamiento condicional con {num_classes} clases")

    # Crear modelo
    model = EnhancedVoxelVAE(
        resolution=args.resolution,
        latent_dim=args.latent_dim,
        conditional=conditional,
        num_classes=num_classes
    ).to(device)

    # Crear optimizador con programación de tasa de aprendizaje
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Historial de pérdidas
    train_losses = []
    bce_losses = []
    kld_losses = []
    perceptual_losses = []

    # Crear directorio para checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.output_dir, f"vae_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Guardar configuración
    config = vars(args)
    config['num_classes'] = num_classes
    config['conditional'] = conditional

    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Entrenar modelo
    print(f"Iniciando entrenamiento por {args.epochs} épocas...")

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_bce = 0
        epoch_kld = 0
        epoch_perceptual = 0

        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch + 1}/{args.epochs}")

        for batch in progress_bar:
            # Obtener datos y moverlos al dispositivo
            voxels = batch['voxels'].to(device)

            # Obtener etiquetas si es condicional
            c = None
            if conditional and 'label_idx' in batch:
                c = torch.tensor(batch['label_idx'], dtype=torch.long).to(device)

            # Reset del gradiente
            optimizer.zero_grad()

            # Forward pass
            recon_voxels, mu, log_var = model(voxels, c)

            # Calcular pérdida
            loss, bce, kld, perceptual = enhanced_vae_loss(
                recon_voxels, voxels, mu, log_var,
                beta=args.beta, perceptual_weight=args.perceptual_weight
            )

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            # Actualizar pérdidas
            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_kld += kld.item()
            epoch_perceptual += perceptual.item()

            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'loss': loss.item() / voxels.size(0),
                'bce': bce.item() / voxels.size(0),
                'kld': kld.item() / voxels.size(0)
            })

        # Calcular pérdidas promedio
        avg_loss = epoch_loss / len(train_dataset)
        avg_bce = epoch_bce / len(train_dataset)
        avg_kld = epoch_kld / len(train_dataset)
        avg_perceptual = epoch_perceptual / len(train_dataset)

        # Actualizar scheduler
        scheduler.step(avg_loss)

        # Guardar historial
        train_losses.append(avg_loss)
        bce_losses.append(avg_bce)
        kld_losses.append(avg_kld)
        perceptual_losses.append(avg_perceptual)

        # Registrar métricas en wandb
        if args.use_wandb:
            wandb.log({
                'loss': avg_loss,
                'bce_loss': avg_bce,
                'kld_loss': avg_kld,
                'perceptual_loss': avg_perceptual,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        # Imprimir estadísticas
        print(f"Época {epoch + 1}/{args.epochs} - Pérdida: {avg_loss:.4f} - BCE: {avg_bce:.4f} - "
              f"KLD: {avg_kld:.4f} - Perceptual: {avg_perceptual:.4f}")

        # Guardar checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint guardado en {checkpoint_path}")

        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, best_model_path)
            print(f"Mejor modelo guardado en {best_model_path}")

        # Generar y guardar algunas muestras
        if args.save_samples and (epoch + 1) % 5 == 0:
            samples_dir = os.path.join(checkpoint_dir, f"samples_epoch_{epoch + 1}")
            os.makedirs(samples_dir, exist_ok=True)

            with torch.no_grad():
                # Generar para cada clase si es condicional
                if conditional and num_classes > 0:
                    for class_idx in range(num_classes):
                        c = torch.tensor([class_idx], device=device)
                        samples = model.sample(num_samples=4, device=device, c=c)

                        for i, sample in enumerate(samples):
                            np.save(os.path.join(samples_dir, f"sample_class_{class_idx}_{i}.npy"),
                                    sample.cpu().numpy())
                else:
                    samples = model.sample(num_samples=8, device=device)

                    for i, sample in enumerate(samples):
                        np.save(os.path.join(samples_dir, f"sample_{i}.npy"),
                                sample.cpu().numpy())

                # Registrar muestras en wandb
                if args.use_wandb:
                    if conditional and num_classes > 0:
                        for class_idx in range(min(4, num_classes)):
                            c = torch.tensor([class_idx], device=device)
                            samples = model.sample(num_samples=1, device=device, c=c)
                            sample_np = samples[0, 0].cpu().numpy()
                            wandb.log({f"sample_class_{class_idx}": wandb.Image(sample_np)})
                    else:
                        samples = model.sample(num_samples=1, device=device)
                        sample_np = samples[0, 0].cpu().numpy()
                        wandb.log({"sample": wandb.Image(sample_np)})

    # Guardar modelo final
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_model_path)
    print(f"Modelo final guardado en {final_model_path}")

    # Guardar gráfica de pérdidas
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Total Loss')
    plt.plot(bce_losses, label='BCE Loss')
    plt.plot(kld_losses, label='KLD Loss')
    plt.plot(perceptual_losses, label='Perceptual Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, "training_loss.png"))
    plt.close()

    # Finalizar wandb
    if args.use_wandb:
        wandb.finish()

    return model, checkpoint_dir


def train_gan(args):
    """
    Entrena la GAN mejorada con seguimiento de experimentos.
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Usando dispositivo: {device}")

    # Inicializar wandb para seguimiento
    if args.use_wandb:
        run = wandb.init(project="decorai-gan", name=f"gan-{args.resolution}-{datetime.now().strftime('%m%d-%H%M')}")
        wandb.config.update(args)

    # Crear dataloader
    train_dataloader, train_dataset = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        resolution=args.resolution,
        category_filter=args.categories.split(',') if args.categories else None
    )

    # Preparar condicionamiento
    num_classes = None
    condition_attributes = None

    if args.conditional:
        # Contar clases únicas
        unique_classes = set()
        for batch in train_dataloader:
            if 'label_idx' in batch:
                unique_classes.update(batch['label_idx'])

        num_classes = len(unique_classes)
        print(f"Entrenamiento condicional con {num_classes} clases")

        # Añadir atributos adicionales si se especifican
        if args.condition_attributes:
            condition_attributes = {}
            for attr_spec in args.condition_attributes.split(','):
                attr_parts = attr_spec.split(':')
                if len(attr_parts) == 2:
                    attr_name, attr_values = attr_parts
                    condition_attributes[attr_name] = int(attr_values)

    # Crear modelo GAN
    model = EnhancedWGAN_GP(
        latent_dim=args.latent_dim,
        channels=args.channels,
        voxel_size=args.resolution,
        num_classes=num_classes,
        condition_attributes=condition_attributes
    ).to(device)

    # Crear optimizadores
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # Crear directorio para checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.output_dir, f"gan_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Configuración
    config = vars(args)
    config['num_classes'] = num_classes

    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Contadores y métricas
    num_train_steps = 0
    d_losses = []
    g_losses = []
    real_scores = []
    fake_scores = []

    # Número de pasos de generador por cada paso de discriminador
    n_critic = args.n_critic

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

            # Preparar etiquetas si es condicional
            labels = None
            if args.conditional and 'label_idx' in batch:
                labels = torch.tensor(batch['label_idx'], dtype=torch.long).to(device)

            # ----------------------
            # Entrenar Discriminador
            # ----------------------

            for _ in range(n_critic):
                optimizer_d.zero_grad()

                # Generar muestras falsas
                g_loss, fake_voxels, _ = model.train_step_generator(batch_size, device, labels)

                # Entrenar discriminador
                if args.conditional:
                    d_loss, real_score, fake_score = model.train_step_discriminator(
                        real_voxels, fake_voxels, lambda_gp=args.lambda_gp,
                        device=device, class_labels=labels
                    )
                else:
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
                g_loss, _, _ = model.train_step_generator(batch_size, device, labels)
            else:
                g_loss, _, _ = model.train_step_generator(batch_size, device)

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

            # Registrar métricas en wandb
            if args.use_wandb and num_train_steps % 50 == 0:
                wandb.log({
                    'step': num_train_steps,
                    'd_loss': d_loss.item(),
                    'g_loss': g_loss.item(),
                    'real_score': real_score.mean().item(),
                    'fake_score': fake_score.mean().item()
                })

            # Guardar muestras periódicamente
            if num_train_steps % args.sample_interval == 0:
                samples_dir = os.path.join(checkpoint_dir, f"samples_step_{num_train_steps}")
                os.makedirs(samples_dir, exist_ok=True)

                with torch.no_grad():
                    if args.conditional and num_classes is not None:
                        # Generar muestras para cada clase
                        for class_idx in range(min(5, num_classes)):
                            class_labels = torch.full((4,), class_idx, dtype=torch.long).to(device)
                            samples, _ = model.generate_samples(4, class_labels, device)

                            for i, sample in enumerate(samples):
                                sample_np = sample.cpu().numpy()
                                np.save(os.path.join(samples_dir, f"sample_class_{class_idx}_{i}.npy"), sample_np)

                            # Registrar en wandb
                            if args.use_wandb:
                                sample_np = samples[0, 0].cpu().numpy()
                                wandb.log({f"sample_class_{class_idx}": wandb.Image(sample_np)})
                    else:
                        # Generar muestras aleatorias
                        samples, _ = model.generate_samples(8, device=device)

                        for i, sample in enumerate(samples):
                            sample_np = sample.cpu().numpy()
                            np.save(os.path.join(samples_dir, f"sample_{i}.npy"), sample_np)

                        # Registrar en wandb
                        if args.use_wandb:
                            sample_np = samples[0, 0].cpu().numpy()
                            wandb.log({"sample": wandb.Image(sample_np)})

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

        # Registrar métricas en wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch_d_loss': epoch_d_loss,
                'epoch_g_loss': epoch_g_loss,
                'epoch_real_score': epoch_real_score,
                'epoch_fake_score': epoch_fake_score
            })

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
                'g_loss': epoch_g_loss,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint guardado en {checkpoint_path}")

    # Guardar modelo final
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'config': config
    }, final_model_path)
    print(f"Modelo final guardado en {final_model_path}")

    # Guardar gráficas de pérdidas
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss')

    plt.subplot(2, 1, 2)
    plt.plot(real_scores, label='Real Scores')
    plt.plot(fake_scores, label='Fake Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Real and Fake Scores')

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "training_plots.png"))
    plt.close()

    # Finalizar wandb
    if args.use_wandb:
        wandb.finish()

    return model, checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento mejorado de modelos generativos 3D")

    parser.add_argument('--model_type', type=str, required=True, choices=['vae', 'gan'],
                        help='Tipo de modelo a entrenar')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directorio de datos de entrenamiento')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directorio de salida para modelos')

    # Parámetros generales
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolución del modelo')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Dimensión del espacio latente')
    parser.add_argument('--conditional', action='store_true',
                        help='Habilitar entrenamiento condicional')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frecuencia de guardado de checkpoints')
    parser.add_argument('--save_samples', action='store_true',
                        help='Guardar muestras durante entrenamiento')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Desactivar CUDA')
    parser.add_argument('--categories', type=str, default=None,
                        help='Categorías a usar (separadas por coma)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Usar Weights & Biases para seguimiento')

    # Parámetros de VAE
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Peso KL en VAE')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Tasa de aprendizaje para VAE')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Peso de pérdida perceptual en VAE')

    # Parámetros de GAN
    parser.add_argument('--channels', type=int, default=64,
                        help='Número base de canales para GAN')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Tasa de aprendizaje del generador')
    parser.add_argument('--lr_d', type=float, default=0.0001,
                        help='Tasa de aprendizaje del discriminador')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Beta1 para Adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='Beta2 para Adam')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Pasos de discriminador por cada paso de generador')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Coeficiente de penalización de gradiente')
    parser.add_argument('--sample_interval', type=int, default=200,
                        help='Intervalo para guardar muestras')
    parser.add_argument('--condition_attributes', type=str, default=None,
                        help='Atributos adicionales para GAN condicional (formato: nombre:valores,...)')

    args = parser.parse_args()

    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)

    # Ejecutar entrenamiento según el tipo de modelo
    if args.model_type == 'vae':
        model, checkpoint_dir = train_vae(args)
        print(f"Entrenamiento VAE completado. Resultados guardados en {checkpoint_dir}")
    elif args.model_type == 'gan':
        model, checkpoint_dir = train_gan(args)
        print(f"Entrenamiento GAN completado. Resultados guardados en {checkpoint_dir}")
    else:
        print(f"Tipo de modelo no soportado: {args.model_type}")


if __name__ == "__main__":
    main()