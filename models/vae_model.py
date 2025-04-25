import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """Bloque de convolución 3D con normalización y activación."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DeconvBlock(nn.Module):
    """Bloque de deconvolución 3D con normalización y activación."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class VoxelVAE(nn.Module):
    """Autoencoder Variacional para generación de modelos 3D voxelizados."""

    def __init__(self, resolution=32, latent_dim=128):
        """
        Inicializa el VAE.

        Args:
            resolution (int): Resolución del modelo voxelizado (debe ser potencia de 2)
            latent_dim (int): Dimensión del espacio latente
        """
        super(VoxelVAE, self).__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim

        # Calcular el número de capas basado en la resolución
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
            encoder_layers.append(
                Conv3DBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # Duplicar canales hasta máx 512

        self.encoder_layers = nn.ModuleList(encoder_layers)

        # Dimensión de la representación aplanada
        self.flatten_dim = in_channels * (self.min_dim ** 3)

        # Capas para VAE (mu y log_var)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Capa para pasar del espacio latente a la representación 3D
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder
        decoder_layers = []
        in_channels = in_channels  # Usar el mismo número de canales del final del encoder

        for i in range(self.num_layers):
            out_channels = in_channels // 2 if i < self.num_layers - 1 else 1
            out_channels = max(out_channels, 16)  # Al menos 16 canales excepto la capa final

            if i == self.num_layers - 1:
                out_channels = 1  # Último layer siempre con 1 canal

            decoder_layers.append(
                DeconvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=1)
            )
            in_channels = out_channels

        self.decoder_layers = nn.ModuleList(decoder_layers)

        # Capa final para normalizar la salida a [0, 1]
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        """Codifica la entrada en distribuciones del espacio latente."""
        batch_size = x.size(0)

        # Pasar por capas del encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Aplanar
        x = x.view(batch_size, -1)

        # Proyectar a espacio latente
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Truco de reparametrización para poder hacer backpropagation."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decodifica del espacio latente a la representación voxelizada."""
        batch_size = z.size(0)

        # Proyectar desde espacio latente
        x = self.fc_decode(z)

        # Reshape a formato 3D
        x = x.view(batch_size, -1, self.min_dim, self.min_dim, self.min_dim)

        # Pasar por capas del decoder
        for layer in self.decoder_layers:
            x = layer(x)

        # Activación final
        x = self.final_activation(x)

        return x

    def forward(self, x):
        """Forward pass completo."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)

        return x_recon, mu, log_var

    def sample(self, num_samples=1, device="cuda"):
        """Genera muestras aleatorias del modelo."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def interpolate(self, x1, x2, steps=10):
        """Interpola entre dos modelos en el espacio latente."""
        mu1, _ = self.encode(x1.unsqueeze(0))
        mu2, _ = self.encode(x2.unsqueeze(0))

        # Crear puntos interpolados en el espacio latente
        ratios = torch.linspace(0, 1, steps=steps)
        vectors = []

        for ratio in ratios:
            z = mu1 * (1 - ratio) + mu2 * ratio
            vectors.append(z)

        z_interp = torch.cat(vectors, dim=0)
        interpolations = self.decode(z_interp)

        return interpolations


# Función de pérdida para VAE
def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    """
    Función de pérdida para VAE: reconstrucción + KL divergence.

    Args:
        recon_x: Salida reconstruida
        x: Entrada original
        mu: Media del espacio latente
        log_var: Log-varianza del espacio latente
        beta: Factor de ponderación para el término KL

    Returns:
        Pérdida total, pérdida de reconstrucción, pérdida KL
    """
    # Pérdida de reconstrucción (BCE para datos binarios)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Pérdida total
    loss = BCE + beta * KLD

    return loss, BCE, KLD


# Ejemplo de uso
if __name__ == "__main__":
    # Dimensiones para prueba
    batch_size = 4
    resolution = 32
    latent_dim = 128

    # Crear modelo
    model = VoxelVAE(resolution=resolution, latent_dim=latent_dim)

    # Entrada de prueba
    x = torch.rand(batch_size, 1, resolution, resolution, resolution)

    # Forward pass
    recon_x, mu, log_var = model(x)

    # Calcular pérdida
    loss, bce, kld = vae_loss(recon_x, x, mu, log_var)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {recon_x.shape}")
    print(f"Total loss: {loss.item()}")
    print(f"BCE: {bce.item()}")
    print(f"KLD: {kld.item()}")