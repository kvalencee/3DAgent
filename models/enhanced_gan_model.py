import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralNorm:
    """Normalización espectral para estabilizar entrenamiento de GAN."""

    def __init__(self):
        self._initialized = False

    def compute_weight(self, module, do_power_iteration):
        if not self._initialized:
            self._initialize(module)

        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(1):  # Menos iteraciones para eficiencia
                    # Power iteration para el valor singular más grande
                    v = self._l2normalize(torch.matmul(weight.view(weight.shape[0], -1).t(), u))
                    u = self._l2normalize(torch.matmul(weight.view(weight.shape[0], -1), v))

                sigma = torch.sum(u * torch.matmul(weight.view(weight.shape[0], -1), v))
                setattr(module, self.name + '_u', u)
                setattr(module, self.name + '_v', v)
        else:
            sigma = torch.sum(u * torch.matmul(weight.view(weight.shape[0], -1), v))

        # Aplicar normalización
        weight_sn = weight / sigma
        setattr(module, self.name, weight_sn)

    def _initialize(self, module):
        weight = getattr(module, self.name)

        u = torch.randn(weight.shape[0], device=weight.device)
        v = torch.randn(weight.view(weight.shape[0], -1).shape[1], device=weight.device)

        u = self._l2normalize(u)
        v = self._l2normalize(v)

        setattr(module, self.name + '_orig', weight.data.clone())
        setattr(module, self.name + '_u', u)
        setattr(module, self.name + '_v', v)

        self._initialized = True

    def _l2normalize(self, x):
        return F.normalize(x, p=2, dim=0)

    def __call__(self, module, inputs):
        self.compute_weight(module, do_power_iteration=module.training)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm()

        weight = getattr(module, name)
        del module._parameters[name]

        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_buffer(name + '_u', torch.randn(weight.shape[0], device=weight.device))
        module.register_buffer(name + '_v',
                               torch.randn(weight.view(weight.shape[0], -1).shape[1], device=weight.device))

        module.register_forward_pre_hook(fn)

        fn.name = name
        return fn


def spectral_norm(module, name='weight'):
    """Aplica normalización espectral a un módulo."""
    SpectralNorm.apply(module, name)
    return module


class ResidualBlock3D(nn.Module):
    """Bloque residual 3D para arquitecturas más profundas."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()

        self.conv1 = spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        self.conv2 = spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride))

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)

        out += self.shortcut(residual)
        out = self.activation(out)

        return out


class EnhancedGenerator(nn.Module):
    """Generador mejorado con soporte para múltiples condiciones."""

    def __init__(self, latent_dim=200, channels=64, voxel_size=64, num_classes=None,
                 class_embedding_dim=64, condition_attributes=None):
        """
        Inicializa el generador mejorado.

        Args:
            latent_dim: Dimensión del espacio latente
            channels: Número base de canales
            voxel_size: Tamaño de salida (resolución)
            num_classes: Número de clases para condicionamiento (opcional)
            class_embedding_dim: Dimensión de embedding para clases
            condition_attributes: Diccionario con atributos adicionales {nombre: num_valores}
        """
        super(EnhancedGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.voxel_size = voxel_size
        self.num_classes = num_classes

        # Calcular el número de capas
        self.num_layers = int(np.log2(voxel_size)) - 2  # 64 -> 4 capas
        self.initial_size = voxel_size // (2 ** self.num_layers)  # 64 -> 4

        # Dimensión condicional total
        self.conditional_dim = 0

        # Embedding para clases
        if num_classes is not None and num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
            self.conditional_dim += class_embedding_dim

        # Embeddings para atributos adicionales
        self.attribute_embeddings = nn.ModuleDict()
        if condition_attributes is not None:
            for attr_name, attr_values in condition_attributes.items():
                self.attribute_embeddings[attr_name] = nn.Embedding(attr_values, class_embedding_dim)
                self.conditional_dim += class_embedding_dim

        # Capa inicial
        initial_channels = channels * (2 ** self.num_layers)

        self.initial_linear = nn.Linear(latent_dim + self.conditional_dim,
                                        initial_channels * (self.initial_size ** 3))

        # Crear capas de deconvolución
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = channels * (2 ** (self.num_layers - i))
            out_channels = channels * (2 ** (self.num_layers - i - 1))

            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock3D(out_channels, out_channels)
            )

            self.layers.append(layer)

        # Capa final
        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, class_labels=None, **condition_attrs):
        """
        Forward pass del generador.

        Args:
            z: Vector latente [batch_size, latent_dim]
            class_labels: Etiquetas de clase (opcional) [batch_size]
            condition_attrs: Atributos condicionales adicionales {nombre: tensor}

        Returns:
            Voxel generado [batch_size, 1, voxel_size, voxel_size, voxel_size]
        """
        batch_size = z.size(0)

        # Concatenar condicionamientos
        if self.num_classes is not None and class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            z = torch.cat([z, class_emb], dim=1)

        # Añadir atributos adicionales
        for attr_name, attr_values in condition_attrs.items():
            if attr_name in self.attribute_embeddings:
                attr_emb = self.attribute_embeddings[attr_name](attr_values)
                z = torch.cat([z, attr_emb], dim=1)

        # Proyectar y reshape
        x = self.initial_linear(z)
        x = x.view(batch_size, -1, self.initial_size, self.initial_size, self.initial_size)

        # Aplicar capas deconv
        for layer in self.layers:
            x = layer(x)

        # Capa final
        x = self.final_layer(x)

        return x


class EnhancedDiscriminator(nn.Module):
    """Discriminador mejorado con soporte para múltiples condiciones."""

    def __init__(self, channels=64, voxel_size=64, num_classes=None,
                 class_embedding_dim=64, condition_attributes=None):
        """
        Inicializa el discriminador mejorado.

        Args:
            channels: Número base de canales
            voxel_size: Tamaño de entrada (resolución)
            num_classes: Número de clases para condicionamiento (opcional)
            class_embedding_dim: Dimensión de embedding para clases
            condition_attributes: Diccionario con atributos adicionales {nombre: num_valores}
        """
        super(EnhancedDiscriminator, self).__init__()

        self.channels = channels
        self.voxel_size = voxel_size
        self.num_classes = num_classes

        # Calcular el número de capas
        self.num_layers = int(np.log2(voxel_size)) - 2  # 64 -> 4 capas

        # Dimensión condicional total
        self.conditional_dim = 0

        # Embedding para clases
        if num_classes is not None and num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
            self.conditional_dim += class_embedding_dim

        # Embeddings para atributos adicionales
        self.attribute_embeddings = nn.ModuleDict()
        if condition_attributes is not None:
            for attr_name, attr_values in condition_attributes.items():
                self.attribute_embeddings[attr_name] = nn.Embedding(attr_values, class_embedding_dim)
                self.conditional_dim += class_embedding_dim

        # Capa inicial
        self.initial_layer = spectral_norm(nn.Conv3d(1, channels, kernel_size=3, stride=1, padding=1))

        # Crear capas de convolución
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = channels * (2 ** i)
            out_channels = channels * (2 ** (i + 1))

            layer = nn.Sequential(
                spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock3D(out_channels, out_channels)
            )

            self.layers.append(layer)

        # Capa final
        final_channels = channels * (2 ** self.num_layers)
        self.final_size = voxel_size // (2 ** self.num_layers)  # 64 -> 4
        self.final_linear = spectral_norm(nn.Linear(final_channels * (self.final_size ** 3) + self.conditional_dim, 1))

    def forward(self, x, class_labels=None, **condition_attrs):
        """
        Forward pass del discriminador.

        Args:
            x: Voxel [batch_size, 1, voxel_size, voxel_size, voxel_size]
            class_labels: Etiquetas de clase (opcional) [batch_size]
            condition_attrs: Atributos condicionales adicionales {nombre: tensor}

        Returns:
            Puntuación de discriminación [batch_size, 1]
        """
        batch_size = x.size(0)

        # Aplicar capa inicial
        x = self.initial_layer(x)

        # Aplicar capas conv
        for layer in self.layers:
            x = layer(x)

        # Aplanar
        x = x.view(batch_size, -1)

        # Concatenar condicionamientos
        if self.num_classes is not None and class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            x = torch.cat([x, class_emb], dim=1)

        # Añadir atributos adicionales
        for attr_name, attr_values in condition_attrs.items():
            if attr_name in self.attribute_embeddings:
                attr_emb = self.attribute_embeddings[attr_name](attr_values)
                x = torch.cat([x, attr_emb], dim=1)

        # Capa final
        x = self.final_linear(x)

        return x


class EnhancedWGAN_GP(nn.Module):
    """GAN mejorada con penalización de gradiente Wasserstein."""

    def __init__(self, latent_dim=200, channels=64, voxel_size=64, num_classes=None,
                 condition_attributes=None):
        """
        Inicializa la GAN mejorada.

        Args:
            latent_dim: Dimensión del espacio latente
            channels: Número base de canales
            voxel_size: Tamaño de los voxels
            num_classes: Número de clases para condicionamiento (opcional)
            condition_attributes: Diccionario con atributos adicionales {nombre: num_valores}
        """
        super(EnhancedWGAN_GP, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.condition_attributes = condition_attributes

        # Crear generador y discriminador
        self.generator = EnhancedGenerator(
            latent_dim=latent_dim,
            channels=channels,
            voxel_size=voxel_size,
            num_classes=num_classes,
            condition_attributes=condition_attributes
        )

        self.discriminator = EnhancedDiscriminator(
            channels=channels,
            voxel_size=voxel_size,
            num_classes=num_classes,
            condition_attributes=condition_attributes
        )

    def generate_samples(self, num_samples, class_labels=None, device='cuda', **condition_attrs):
        """
        Genera muestras aleatorias.

        Args:
            num_samples: Número de muestras a generar
            class_labels: Etiquetas de clase (opcional)
            device: Dispositivo (CPU/GPU)
            condition_attrs: Atributos condicionales adicionales

        Returns:
            Voxels generados, vectores latentes
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)

        with torch.no_grad():
            samples = self.generator(z, class_labels, **condition_attrs)

        return samples, z

    def train_step_generator(self, batch_size, device, class_labels=None, **condition_attrs):
        """
        Paso de entrenamiento para el generador.
        """
        # Generar ruido aleatorio
        z = torch.randn(batch_size, self.latent_dim).to(device)

        # Generar muestras falsas
        fake_voxels = self.generator(z, class_labels, **condition_attrs)

        # Calcular puntuación del discriminador para muestras falsas
        fake_score = self.discriminator(fake_voxels, class_labels, **condition_attrs)

        # Pérdida del generador (maximizar puntuación)
        g_loss = -torch.mean(fake_score)

        return g_loss, fake_voxels, z

    def train_step_discriminator(self, real_voxels, fake_voxels, lambda_gp=10.0, device='cuda',
                                 class_labels=None, fake_labels=None, **condition_attrs):
        """
        Paso de entrenamiento para el discriminador.
        """
        batch_size = real_voxels.size(0)

        # Calcular puntuación para muestras reales
        real_score = self.discriminator(real_voxels, class_labels, **condition_attrs)

        # Calcular puntuación para muestras falsas
        fake_labels_to_use = fake_labels if fake_labels is not None else class_labels
        fake_score = self.discriminator(fake_voxels.detach(), fake_labels_to_use, **condition_attrs)

        # Calcular pérdida Wasserstein
        d_loss = torch.mean(fake_score) - torch.mean(real_score)

        # Calcular penalización de gradiente
        alpha = torch.rand(batch_size, 1, 1, 1, 1).to(device)
        interpolated = (alpha * real_voxels + (1 - alpha) * fake_voxels.detach()).requires_grad_(True)

        # Calcular puntuación para muestras interpoladas
        interpolated_score = self.discriminator(interpolated, class_labels, **condition_attrs)

        # Calcular gradientes
        gradients = torch.autograd.grad(
            outputs=interpolated_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_score).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Aplanar gradientes
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Añadir penalización a la pérdida
        d_loss = d_loss + lambda_gp * gradient_penalty

        return d_loss, real_score, fake_score