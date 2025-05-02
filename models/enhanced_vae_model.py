import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """Bloque residual 3D para arquitecturas más profundas."""

    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Conexión residual
        out = self.activation(out)
        return out


class EnhancedVoxelVAE(nn.Module):
    """VAE mejorado con arquitectura más profunda y opciones condicionales."""

    def __init__(self, resolution=64, latent_dim=128, conditional=False, num_classes=0):
        """
        Inicializa el VAE mejorado.

        Args:
            resolution: Resolución del modelo voxelizado
            latent_dim: Dimensión del espacio latente
            conditional: Si es condicional en clases
            num_classes: Número de clases si es condicional
        """
        super(EnhancedVoxelVAE, self).__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes

        # Calcular el número de capas
        self.num_layers = 0
        temp_res = resolution
        while temp_res > 4:  # Reducir hasta 4x4x4
            self.num_layers += 1
            temp_res = temp_res // 2

        # Dimensiones de la capa más profunda
        self.min_dim = resolution // (2 ** self.num_layers)

        # Encoder
        encoder_layers = []
        in_channels = 1  # Comenzando con 1 canal (voxel binario)
        out_channels = 32  # Primer número de canales de salida

        for i in range(self.num_layers):
            # Bloque convolucional
            encoder_layers.append(
                nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            encoder_layers.append(nn.BatchNorm3d(out_channels))
            encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))

            # Añadir bloque residual para capas intermedias
            if i > 0 and i < self.num_layers - 1:
                encoder_layers.append(ResidualBlock3D(out_channels))

            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # Duplicar canales hasta máx 512

        self.encoder = nn.Sequential(*encoder_layers)

        # Dimensión de la representación aplanada
        self.flatten_dim = in_channels * (self.min_dim ** 3)

        # Capas para condicionamiento (si es condicional)
        if conditional and num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, 64)
            self.fc_combine = nn.Linear(self.flatten_dim + 64, self.flatten_dim)

        # Capas para VAE (mu y log_var)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Capa para pasar del espacio latente a la representación 3D
        if conditional and num_classes > 0:
            self.fc_decode = nn.Linear(latent_dim + 64, self.flatten_dim)
        else:
            self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder
        decoder_layers = []
        in_channels = in_channels  # Usar el mismo número de canales del final del encoder

        # Reshape a formato 3D
        self.decoder_input = nn.Linear(self.flatten_dim, in_channels * self.min_dim ** 3)

        for i in range(self.num_layers):
            out_channels = in_channels // 2 if i < self.num_layers - 1 else 1
            out_channels = max(out_channels, 16)  # Al menos 16 canales excepto la capa final

            if i == self.num_layers - 1:
                out_channels = 1  # Último layer siempre con 1 canal

            # Bloque de deconvolución
            decoder_layers.append(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )

            if i < self.num_layers - 1:
                decoder_layers.append(nn.BatchNorm3d(out_channels))
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))

                # Añadir bloque residual para capas intermedias
                if i > 0:
                    decoder_layers.append(ResidualBlock3D(out_channels))
            else:
                # Capa final con activación sigmoid
                decoder_layers.append(nn.Sigmoid())

            in_channels = out_channels

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, c=None):
        """Codifica la entrada en distribuciones del espacio latente."""
        batch_size = x.size(0)

        # Pasar por capas del encoder
        x = self.encoder(x)

        # Aplanar
        x = x.view(batch_size, -1)

        # Condicionamiento (si es condicional)
        if self.conditional and c is not None:
            c_emb = self.class_embedding(c)
            x = torch.cat([x, c_emb], dim=1)
            x = self.fc_combine(x)

        # Proyectar a espacio latente
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Truco de reparametrización para backpropagation."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, c=None):
        """Decodifica del espacio latente a la representación voxelizada."""
        batch_size = z.size(0)

        # Condicionamiento (si es condicional)
        if self.conditional and c is not None:
            c_emb = self.class_embedding(c)
            z = torch.cat([z, c_emb], dim=1)

        # Proyectar desde espacio latente
        x = self.fc_decode(z)

        # Reshape a formato 3D
        x = x.view(batch_size, -1, self.min_dim, self.min_dim, self.min_dim)

        # Pasar por capas del decoder
        x = self.decoder(x)

        return x

    def forward(self, x, c=None):
        """Forward pass completo."""
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)

        return x_recon, mu, log_var

    def sample(self, num_samples=1, device="cuda", c=None):
        """Genera muestras aleatorias del modelo."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z, c)
        return samples

    def interpolate(self, x1, x2, steps=10, c=None):
        """Interpola entre dos modelos en el espacio latente."""
        mu1, _ = self.encode(x1.unsqueeze(0), c)
        mu2, _ = self.encode(x2.unsqueeze(0), c)

        # Crear puntos interpolados en el espacio latente
        vectors = []
        for t in torch.linspace(0, 1, steps=steps):
            z = mu1 * (1 - t) + mu2 * t
            vectors.append(z)

        z_interp = torch.cat(vectors, dim=0)

        # Mismo condicionamiento para todas las interpolaciones
        if c is not None:
            c_interp = c.repeat(steps)
        else:
            c_interp = None

        interpolations = self.decode(z_interp, c_interp)

        return interpolations


# Función de pérdida para VAE con pérdida perceptual
def enhanced_vae_loss(recon_x, x, mu, log_var, beta=1.0, perceptual_weight=0.1):
    """
    Función de pérdida mejorada para VAE.

    Args:
        recon_x: Salida reconstruida
        x: Entrada original
        mu: Media del espacio latente
        log_var: Log-varianza del espacio latente
        beta: Factor de ponderación para el término KL
        perceptual_weight: Peso para la pérdida perceptual

    Returns:
        Pérdida total, pérdida de reconstrucción, pérdida KL, pérdida perceptual
    """
    # Pérdida de reconstrucción (BCE para datos binarios)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Pérdida perceptual (diferencia entre características)
    # Implementamos una versión simplificada basada en gradientes
    perceptual_loss = 0.0
    if perceptual_weight > 0:
        # Gradientes en X, Y, Z
        grad_x = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        grad_y = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        grad_z = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

        grad_x_recon = torch.abs(recon_x[:, :, 1:, :, :] - recon_x[:, :, :-1, :, :])
        grad_y_recon = torch.abs(recon_x[:, :, :, 1:, :] - recon_x[:, :, :, :-1, :])
        grad_z_recon = torch.abs(recon_x[:, :, :, :, 1:] - recon_x[:, :, :, :, :-1])

        # Diferencia de gradientes (preserva bordes/estructuras)
        perceptual_loss = F.mse_loss(grad_x, grad_x_recon, reduction='sum') + \
                          F.mse_loss(grad_y, grad_y_recon, reduction='sum') + \
                          F.mse_loss(grad_z, grad_z_recon, reduction='sum')

    # Pérdida total
    loss = BCE + beta * KLD + perceptual_weight * perceptual_loss

    return loss, BCE, KLD, perceptual_loss