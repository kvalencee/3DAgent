import os
import sys
import argparse
import numpy as np
import trimesh
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import json

# Añadir la carpeta raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Funciones para crear formas básicas paramétrizadas

def create_vase(height=1.0, radius_top=0.3, radius_bottom=0.5, num_sections=5,
                resolution=32, noise_amount=0.0, smooth=True, hollow=True,
                wall_thickness=0.05, bottom_thickness=0.05, neck_position=None, neck_radius_factor=None):
    """
    Crea un modelo 3D de un florero con parámetros personalizables.

    Args:
        ... (parámetros existentes) ...
        neck_position: Posición relativa del cuello (entre 0 y 1) o None
        neck_radius_factor: Factor de reducción del radio en el cuello
    """
    # Crear puntos de perfil para el florero
    sections = np.linspace(0, 1, num_sections)
    radii = []

    # Definir función de interpolación de radio
    for t in sections:
        # Interpolación básica entre radio inferior y superior
        r = radius_bottom + (radius_top - radius_bottom) * t

        # Añadir cuello si se especifica
        if neck_position is not None and neck_radius_factor is not None:
            # Crear un estrechamiento en la posición del cuello
            neck_effect = np.exp(-10 * (t - neck_position) ** 2) * (1 - neck_radius_factor)
            r = r * (1 - neck_effect)

        # Función sinusoidal para crear curvas interesantes
        r += 0.2 * np.sin(t * np.pi) * (radius_top + radius_bottom) / 2

        # Añadir ruido para más variación
        if noise_amount > 0:
            r += np.random.uniform(-noise_amount, noise_amount) * (radius_top + radius_bottom) / 2

        radii.append(max(0.1, r))  # Asegurar radio positivo

    heights = np.linspace(0, height, num_sections)

    # Crear puntos de perfil
    profile = np.column_stack((radii, heights))

    # Crear malla mediante revolución del perfil
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    vertices = []
    faces = []

    # Generar vértices
    for i, (r, h) in enumerate(profile):
        for j, angle in enumerate(theta):
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = h
            vertices.append([x, y, z])

    # Generar caras conectando vértices
    for i in range(num_sections - 1):
        for j in range(resolution):
            # Índices de los vértices
            v1 = i * resolution + j
            v2 = i * resolution + (j + 1) % resolution
            v3 = (i + 1) * resolution + (j + 1) % resolution
            v4 = (i + 1) * resolution + j

            # Crear dos triángulos para formar un cuadrilátero
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # Crear malla
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    # Añadir base y tapa
    if radius_bottom > 0:
        # Crear base
        base_center = [0, 0, 0]
        base_verts = [base_center]
        base_verts.extend([[r * np.cos(angle), r * np.sin(angle), 0]
                           for r, angle in zip([radius_bottom] * resolution, theta)])

        base_faces = [[0, j + 1, (j + 1) % resolution + 1] for j in range(resolution)]

        base_mesh = trimesh.Trimesh(vertices=np.array(base_verts),
                                    faces=np.array(base_faces))
        mesh = trimesh.util.concatenate([mesh, base_mesh])

    # Si es hueco, crear interior
    if hollow and wall_thickness > 0:
        # Crear perfil interior
        inner_radii = [max(0, r - wall_thickness) for r in radii]
        inner_profile = np.column_stack((inner_radii, heights))

        # Generar interior usando el mismo proceso
        inner_vertices = []
        inner_faces = []

        # Generar vértices interiores
        for i, (r, h) in enumerate(inner_profile):
            if i == 0:  # Ajuste para el fondo
                h += bottom_thickness

            for j, angle in enumerate(theta):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = h
                inner_vertices.append([x, y, z])

        # Generar caras interiores (invertidas)
        for i in range(num_sections - 1):
            for j in range(resolution):
                v1 = i * resolution + j
                v2 = i * resolution + (j + 1) % resolution
                v3 = (i + 1) * resolution + (j + 1) % resolution
                v4 = (i + 1) * resolution + j

                # Invertir el orden para que las normales apunten hacia adentro
                inner_faces.append([v1, v3, v2])
                inner_faces.append([v1, v4, v3])

        # Crear malla interior
        inner_mesh = trimesh.Trimesh(vertices=np.array(inner_vertices),
                                     faces=np.array(inner_faces))

        # Crear borde superior
        top_verts = []
        top_faces = []

        # Añadir vértices del borde
        outer_top_start = (num_sections - 1) * resolution
        inner_top_start = (num_sections - 1) * resolution

        for j in range(resolution):
            top_verts.append(vertices[outer_top_start + j])
            top_verts.append(inner_vertices[inner_top_start + j])

            # Índices para el cuadrilátero del borde
            v1 = j * 2
            v2 = (j * 2 + 2) % (resolution * 2)
            v3 = (j * 2 + 3) % (resolution * 2)
            v4 = j * 2 + 1

            top_faces.append([v1, v2, v3])
            top_faces.append([v1, v3, v4])

        top_mesh = trimesh.Trimesh(vertices=np.array(top_verts),
                                   faces=np.array(top_faces))

        # Unir todas las partes
        mesh = trimesh.util.concatenate([mesh, inner_mesh, top_mesh])

    # Suavizar malla si se solicita
    if smooth:
        mesh = mesh.smoothed()

    return mesh


def create_planter(height=0.8, width=0.7, depth=0.7, rim_thickness=0.08,
                   wall_thickness=0.05, bottom_thickness=0.08, resolution=32,
                   rounded=True, corner_radius=0.1, noise_amount=0.0):
    """
    Crea un modelo 3D de una maceta o plantador.

    Args:
        height: Altura de la maceta
        width: Ancho de la maceta
        depth: Profundidad de la maceta
        rim_thickness: Grosor del borde superior
        wall_thickness: Grosor de las paredes
        bottom_thickness: Grosor del fondo
        resolution: Resolución de la malla
        rounded: Si las esquinas deben ser redondeadas
        corner_radius: Radio de las esquinas redondeadas
        noise_amount: Cantidad de ruido para añadir variación

    Returns:
        trimesh.Trimesh: Malla 3D de la maceta
    """
    # Crear forma exterior
    outer_box = trimesh.creation.box(extents=[width, depth, height])

    # Si se desean esquinas redondeadas
    if rounded and corner_radius > 0:
        # Crear cilindros para las esquinas
        cylinders = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                # Posición del cilindro
                pos_x = x * (width / 2 - corner_radius)
                pos_y = y * (depth / 2 - corner_radius)

                # Crear cilindro vertical
                cylinder = trimesh.creation.cylinder(
                    radius=corner_radius,
                    height=height,
                    sections=resolution // 4
                )

                # Mover a la posición correcta
                cylinder.apply_translation([pos_x, pos_y, height / 2])
                cylinders.append(cylinder)

        # Unir cilindros con la caja
        outer_box = trimesh.boolean.union([outer_box] + cylinders)

    # Crear forma interior (hueco)
    inner_width = width - 2 * wall_thickness
    inner_depth = depth - 2 * wall_thickness
    inner_height = height - bottom_thickness

    inner_box = trimesh.creation.box(
        extents=[inner_width, inner_depth, inner_height]
    )

    # Mover el hueco a la posición correcta
    inner_box.apply_translation([0, 0, (height - inner_height) / 2 + bottom_thickness / 2])

    # Si se desean esquinas redondeadas para el interior
    if rounded and corner_radius > 0:
        inner_corner_radius = max(0, corner_radius - wall_thickness)
        inner_cylinders = []

        for x in [-1, 1]:
            for y in [-1, 1]:
                # Posición del cilindro interior
                pos_x = x * (inner_width / 2 - inner_corner_radius)
                pos_y = y * (inner_depth / 2 - inner_corner_radius)

                # Crear cilindro vertical interior
                cylinder = trimesh.creation.cylinder(
                    radius=inner_corner_radius,
                    height=inner_height,
                    sections=resolution // 4
                )

                # Mover a la posición correcta
                cylinder.apply_translation([pos_x, pos_y, inner_height / 2 + bottom_thickness])
                inner_cylinders.append(cylinder)

        # Unir cilindros con la caja interior
        inner_box = trimesh.boolean.union([inner_box] + inner_cylinders)

    # Hacer hueco en la maceta
    planter = trimesh.boolean.difference(outer_box, inner_box)

    # Crear borde superior si se especifica
    if rim_thickness > 0:
        rim_height = rim_thickness
        rim_width = width + 2 * rim_thickness
        rim_depth = depth + 2 * rim_thickness

        # Crear borde
        rim = trimesh.creation.box(extents=[rim_width, rim_depth, rim_height])

        # Mover el borde a la posición correcta
        rim.apply_translation([0, 0, height - rim_height / 2])

        # Si se desean esquinas redondeadas para el borde
        if rounded and corner_radius > 0:
            rim_corner_radius = corner_radius + rim_thickness
            rim_cylinders = []

            for x in [-1, 1]:
                for y in [-1, 1]:
                    # Posición del cilindro del borde
                    pos_x = x * (rim_width / 2 - rim_corner_radius)
                    pos_y = y * (rim_depth / 2 - rim_corner_radius)

                    # Crear cilindro para el borde
                    cylinder = trimesh.creation.cylinder(
                        radius=rim_corner_radius,
                        height=rim_height,
                        sections=resolution // 4
                    )

                    # Mover a la posición correcta
                    cylinder.apply_translation([pos_x, pos_y, height - rim_height / 2])
                    rim_cylinders.append(cylinder)

            # Unir cilindros con el borde
            rim = trimesh.boolean.union([rim] + rim_cylinders)

        # Unir borde con la maceta
        planter = trimesh.boolean.union([planter, rim])

    # Añadir ruido si se especifica
    if noise_amount > 0:
        vertices = planter.vertices.copy()

        # Añadir ruido a los vértices
        noise = np.random.uniform(-noise_amount, noise_amount, size=vertices.shape)
        vertices += noise

        # Crear nueva malla con vértices ruidosos
        planter = trimesh.Trimesh(vertices=vertices, faces=planter.faces)

    return planter


def create_bowl(radius=0.5, height=0.3, thickness=0.05, resolution=32,
                base_radius=0.25, noise_amount=0.0, smooth=True):
    """
    Crea un modelo 3D de un cuenco.

    Args:
        radius: Radio del cuenco
        height: Altura del cuenco
        thickness: Grosor de las paredes
        resolution: Resolución de la malla
        base_radius: Radio de la base
        noise_amount: Cantidad de ruido para añadir variación
        smooth: Si se debe suavizar la malla

    Returns:
        trimesh.Trimesh: Malla 3D del cuenco
    """
    # Crear una semiesfera para el exterior
    sphere = trimesh.creation.icosphere(radius=radius, subdivisions=3)

    # Cortar la semiesfera
    z_plane = trimesh.creation.box(extents=[radius * 3, radius * 3, radius * 3])
    z_plane.apply_translation([0, 0, -radius + height])

    # Exterior del cuenco
    bowl_exterior = trimesh.boolean.intersection([sphere, z_plane])

    # Crear interior hueco
    sphere_inner = trimesh.creation.icosphere(
        radius=max(radius - thickness, 0.01),
        subdivisions=3
    )

    # Hueco interior (evitar cortar la base)
    z_plane_inner = trimesh.creation.box(extents=[radius * 3, radius * 3, radius * 3])
    z_plane_inner.apply_translation([0, 0, -radius + height + thickness])

    # Interior del cuenco
    bowl_interior = trimesh.boolean.intersection([sphere_inner, z_plane_inner])

    # Crear el cuenco final
    bowl = trimesh.boolean.difference(bowl_exterior, bowl_interior)

    # Crear base plana
    if base_radius > 0:
        # Cilindro plano para la base
        base = trimesh.creation.cylinder(
            radius=base_radius,
            height=thickness / 2,
            sections=resolution
        )

        # Posicionar en la parte inferior
        base.apply_translation([0, 0, thickness / 4])

        # Unir con el cuenco
        bowl = trimesh.boolean.union([bowl, base])

    # Añadir ruido si se especifica
    if noise_amount > 0:
        vertices = bowl.vertices.copy()

        # Añadir ruido a los vértices (preservar la base)
        for i in range(len(vertices)):
            # Menor ruido cerca de la base
            z_factor = min(1.0, vertices[i][2] / height)
            local_noise = noise_amount * z_factor

            noise = np.random.uniform(-local_noise, local_noise, size=3)
            vertices[i] += noise

        # Crear nueva malla con vértices ruidosos
        bowl = trimesh.Trimesh(vertices=vertices, faces=bowl.faces)

    # Suavizar malla si se solicita
    if smooth:
        bowl = bowl.smoothed()

    return bowl


def create_candleholder(height=0.4, base_radius=0.3, top_radius=0.2,
                        candle_radius=0.1, candle_depth=0.15, num_sections=5,
                        resolution=32, noise_amount=0.0, decorative=True):
    """
    Crea un modelo 3D de un portavelas.

    Args:
        height: Altura del portavelas
        base_radius: Radio de la base
        top_radius: Radio de la parte superior
        candle_radius: Radio del hueco para la vela
        candle_depth: Profundidad del hueco para la vela
        num_sections: Número de secciones para la forma
        resolution: Resolución de la malla
        noise_amount: Cantidad de ruido para añadir variación
        decorative: Si se deben añadir elementos decorativos

    Returns:
        trimesh.Trimesh: Malla 3D del portavelas
    """
    # Crear puntos de perfil para el portavelas
    sections = np.linspace(0, 1, num_sections)
    radii = []

    # Definir función de interpolación de radio
    for t in sections:
        # Función para crear perfil interesante
        r = base_radius * (1 - t) + top_radius * t

        # Añadir ondulación si es decorativo
        if decorative:
            r += 0.05 * np.sin(t * np.pi * 4) * (base_radius + top_radius) / 2

        # Añadir ruido para más variación
        if noise_amount > 0:
            r += np.random.uniform(-noise_amount, noise_amount) * (base_radius + top_radius) / 2

        radii.append(max(0.1, r))  # Asegurar radio positivo

    heights = np.linspace(0, height, num_sections)

    # Crear puntos de perfil
    profile = np.column_stack((radii, heights))

    # Crear malla mediante revolución del perfil
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    vertices = []
    faces = []

    # Generar vértices
    for i, (r, h) in enumerate(profile):
        for j, angle in enumerate(theta):
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = h
            vertices.append([x, y, z])

    # Generar caras conectando vértices
    for i in range(num_sections - 1):
        for j in range(resolution):
            # Índices de los vértices
            v1 = i * resolution + j
            v2 = i * resolution + (j + 1) % resolution
            v3 = (i + 1) * resolution + (j + 1) % resolution
            v4 = (i + 1) * resolution + j

            # Crear dos triángulos para formar un cuadrilátero
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # Crear malla
    base_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    # Crear hueco para la vela
    if candle_radius > 0 and candle_depth > 0:
        # Crear cilindro para el hueco
        candle_hole = trimesh.creation.cylinder(
            radius=candle_radius,
            height=candle_depth * 2,  # Más largo para asegurar el corte completo
            sections=resolution
        )

        # Posicionar en la parte superior
        candle_hole.apply_translation([0, 0, height - candle_depth / 2])

        # Hacer el hueco
        base_mesh = trimesh.boolean.difference(base_mesh, candle_hole)

    # Añadir base si se necesita
    if base_radius > top_radius:
        # Crear disco para base
        base_height = height * 0.1  # 10% de la altura total
        base = trimesh.creation.cylinder(
            radius=base_radius,
            height=base_height,
            sections=resolution
        )

        # Posicionar en la parte inferior
        base.apply_translation([0, 0, base_height / 2])

        # Unir con la base
        base_mesh = trimesh.boolean.union([base_mesh, base])

    # Añadir elementos decorativos si se solicita
    if decorative:
        # Añadir patrones ornamentales
        num_ornaments = random.randint(3, 7)
        ornament_height = height * 0.6  # Altura donde colocar ornamentos

        for i in range(num_ornaments):
            angle = i * (2 * np.pi / num_ornaments)

            # Radio en el punto de ornamento (interpolación)
            ornament_t = ornament_height / height
            ornament_r = base_radius * (1 - ornament_t) + top_radius * ornament_t

            # Posición del ornamento
            pos_x = ornament_r * 0.9 * np.cos(angle)
            pos_y = ornament_r * 0.9 * np.sin(angle)

            # Crear esfera para ornamento
            ornament = trimesh.creation.icosphere(
                radius=height * 0.05,
                subdivisions=2
            )

            # Posicionar ornamento
            ornament.apply_translation([pos_x, pos_y, ornament_height])

            # Unir con el portavelas
            base_mesh = trimesh.boolean.union([base_mesh, ornament])

    return base_mesh


def create_decorative_plate(radius=0.5, height=0.1, rim_width=0.08,
                            resolution=32, pattern='simple', noise_amount=0.0):
    """
    Crea un modelo 3D de un plato decorativo.

    Args:
        radius: Radio del plato
        height: Altura/grosor del plato
        rim_width: Ancho del borde
        resolution: Resolución de la malla
        pattern: Patrón decorativo ('simple', 'floral', 'geometric')
        noise_amount: Cantidad de ruido para añadir variación

    Returns:
        trimesh.Trimesh: Malla 3D del plato
    """
    # Crear disco base
    plate = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=resolution
    )

    # Centrar en Z
    plate.apply_translation([0, 0, height / 2])

    # Crear borde elevado
    if rim_width > 0:
        inner_radius = radius - rim_width
        rim_height = height * 1.5

        # Crear anillo para el borde
        inner_cylinder = trimesh.creation.cylinder(
            radius=inner_radius,
            height=height,
            sections=resolution
        )

        inner_cylinder.apply_translation([0, 0, height / 2])

        outer_cylinder = trimesh.creation.cylinder(
            radius=radius,
            height=rim_height,
            sections=resolution
        )

        outer_cylinder.apply_translation([0, 0, rim_height / 2])

        # Crear el borde
        rim = trimesh.boolean.difference(outer_cylinder, inner_cylinder)

        # Unir borde con plato
        plate = trimesh.boolean.union([plate, rim])

    # Añadir patrones decorativos
    if pattern != 'simple':
        num_patterns = resolution // 4  # Número de patrones repetidos

        for i in range(num_patterns):
            angle = i * (2 * np.pi / num_patterns)
            pattern_pos_r = radius - rim_width / 2

            pattern_pos_x = pattern_pos_r * np.cos(angle)
            pattern_pos_y = pattern_pos_r * np.sin(angle)

            if pattern == 'floral':
                # Crear motivo floral simple
                petal_size = rim_width * 0.7
                pattern_mesh = trimesh.creation.icosphere(
                    radius=petal_size,
                    subdivisions=2
                )

                pattern_mesh.apply_translation([pattern_pos_x, pattern_pos_y, height + petal_size * 0.7])

            elif pattern == 'geometric':
                # Crear motivo geométrico
                pattern_height = height * 0.8
                pattern_size = rim_width * 0.5

                pattern_mesh = trimesh.creation.box(
                    extents=[pattern_size, pattern_size, pattern_height]
                )

                # Rotar para apuntar hacia el centro
                angle_rotation = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])

                pattern_mesh.apply_transform(
                    np.vstack([
                        np.hstack([angle_rotation, np.zeros((3, 1))]),
                        [0, 0, 0, 1]
                    ])
                )

                pattern_mesh.apply_translation([pattern_pos_x, pattern_pos_y, height + pattern_height / 2])

            # Unir patrón con plato
            plate = trimesh.boolean.union([plate, pattern_mesh])

    # Añadir ruido si se especifica
    if noise_amount > 0:
        vertices = plate.vertices.copy()

        # Añadir ruido a los vértices (preservar la planitud del centro)
        for i in range(len(vertices)):
            vertex = vertices[i]

            # Calcular distancia desde el centro (xy)
            dist_from_center = np.sqrt(vertex[0] ** 2 + vertex[1] ** 2)

            # Mayor ruido en el borde
            edge_factor = min(1.0, dist_from_center / radius)
            local_noise = noise_amount * edge_factor

            noise = np.random.uniform(-local_noise, local_noise, size=3)
            vertices[i] += noise

        # Crear nueva malla con vértices ruidosos
        plate = trimesh.Trimesh(vertices=vertices, faces=plate.faces)

    return plate


def generate_random_vase():
    """Genera un florero aleatorio con parámetros variados."""
    # Aumentar rangos para más diversidad
    height = np.random.uniform(0.6, 2.5)  # Más variedad en altura
    radius_top = np.random.uniform(0.1, 0.8)  # Más variedad en apertura
    radius_bottom = np.random.uniform(0.2, 0.9)  # Más variedad en base
    num_sections = np.random.randint(4, 20)  # Más secciones para más complejidad
    noise_amount = np.random.uniform(0, 0.08)  # Más textura irregular

    # Añadir ocasionalmente un cuello estrecho
    if np.random.random() < 0.3:  # 30% de probabilidad
        neck_position = np.random.uniform(0.3, 0.7)  # Posición relativa del cuello
        neck_radius_factor = np.random.uniform(0.4, 0.7)  # Qué tan estrecho es el cuello
    else:
        neck_position = None
        neck_radius_factor = None

    return create_vase(
        height=height,
        radius_top=radius_top,
        radius_bottom=radius_bottom,
        num_sections=num_sections,
        noise_amount=noise_amount,
        hollow=np.random.choice([True, False], p=[0.8, 0.2]),
        wall_thickness=np.random.uniform(0.02, 0.1),
        # Pasar los nuevos parámetros
        neck_position=neck_position,
        neck_radius_factor=neck_radius_factor
    )


def generate_random_planter():
    """Genera una maceta aleatoria con parámetros variados."""
    height = np.random.uniform(0.6, 1.2)
    width = np.random.uniform(0.5, 1.0)
    depth = width * np.random.uniform(0.8, 1.2)  # Proporcional al ancho
    rounded = np.random.choice([True, False], p=[0.7, 0.3])

    return create_planter(
        height=height,
        width=width,
        depth=depth,
        rim_thickness=np.random.uniform(0.05, 0.12),
        wall_thickness=np.random.uniform(0.04, 0.08),
        rounded=rounded,
        corner_radius=np.random.uniform(0.05, 0.15) if rounded else 0,
        noise_amount=np.random.uniform(0, 0.03)
    )


def generate_random_bowl():
    """Genera un cuenco aleatorio con parámetros variados."""
    radius = np.random.uniform(0.4, 0.8)
    height = radius * np.random.uniform(0.5, 0.8)

    return create_bowl(
        radius=radius,
        height=height,
        thickness=np.random.uniform(0.03, 0.08),
        base_radius=radius * np.random.uniform(0.3, 0.6),
        noise_amount=np.random.uniform(0, 0.04),
        smooth=np.random.choice([True, False], p=[0.8, 0.2])
    )


def generate_random_candleholder():
    """Genera un portavelas aleatorio con parámetros variados."""
    height = np.random.uniform(0.3, 0.7)
    base_radius = np.random.uniform(0.2, 0.4)
    top_radius = base_radius * np.random.uniform(0.6, 1.0)

    return create_candleholder(
        height=height,
        base_radius=base_radius,
        top_radius=top_radius,
        candle_radius=top_radius * np.random.uniform(0.4, 0.7),
        candle_depth=height * np.random.uniform(0.2, 0.5),
        num_sections=np.random.randint(4, 10),
        noise_amount=np.random.uniform(0, 0.04),
        decorative=np.random.choice([True, False], p=[0.7, 0.3])
    )


def generate_random_plate():
    """Genera un plato decorativo aleatorio con parámetros variados."""
    radius = np.random.uniform(0.4, 0.8)

    return create_decorative_plate(
        radius=radius,
        height=radius * np.random.uniform(0.1, 0.2),
        rim_width=radius * np.random.uniform(0.1, 0.3),
        pattern=np.random.choice(['simple', 'floral', 'geometric']),
        noise_amount=np.random.uniform(0, 0.03)
    )


def voxelize_mesh(mesh, resolution=32):
    """
    Convierte una malla 3D en una representación voxelizada.

    Args:
        mesh: Malla 3D
        resolution: Resolución de la representación voxelizada

    Returns:
        np.ndarray: Representación voxelizada (valores de 0 a 1)
    """
    # Asegurarse de que la malla está centrada y escalada correctamente
    mesh = mesh.copy()

    # Normalizar la malla para que quepa en un cubo unitario
    extents = mesh.bounding_box.extents
    scale = 0.95 / max(extents)  # Dejar un pequeño margen
    mesh.apply_scale(scale)

    # Centrar en el origen
    mesh.apply_translation(-mesh.bounding_box.centroid)

    # Crear representación voxelizada
    voxels = mesh.voxelized(pitch=1.0 / resolution)

    # Obtener matriz binaria
    voxel_grid = voxels.matrix

    # Asegurarse de que la forma es correcta
    if voxel_grid.shape != (resolution, resolution, resolution):
        voxel_grid_resized = np.zeros((resolution, resolution, resolution), dtype=bool)
        min_dims = [min(voxel_grid.shape[i], resolution) for i in range(3)]
        voxel_grid_resized[:min_dims[0], :min_dims[1], :min_dims[2]] = voxel_grid[:min_dims[0], :min_dims[1],
                                                                       :min_dims[2]]
        voxel_grid = voxel_grid_resized

    return voxel_grid.astype(np.float32)


def create_synthetic_dataset(num_samples=100, output_dir="data/synthetic",
                             categories=None, resolution=32, visualize=False):
    """
    Crea un conjunto de datos sintético de modelos 3D decorativos.

    Args:
        num_samples: Número total de muestras a generar
        output_dir: Directorio de salida
        categories: Diccionario con categorías y sus proporciones
        resolution: Resolución de la representación voxelizada
        visualize: Si se deben visualizar los modelos generados

    Returns:
        dict: Estadísticas de generación
    """
    if categories is None:
        # Categorías predeterminadas con sus generadores y proporciones
        categories = {
            "vase": {"generator": generate_random_vase, "proportion": 0.3},
            "planter": {"generator": generate_random_planter, "proportion": 0.2},
            "bowl": {"generator": generate_random_bowl, "proportion": 0.2},
            "candleholder": {"generator": generate_random_candleholder, "proportion": 0.15},
            "plate": {"generator": generate_random_plate, "proportion": 0.15}
        }

    # Normalizar proporciones
    total_proportion = sum(cat["proportion"] for cat in categories.values())
    for cat in categories.values():
        cat["proportion"] /= total_proportion

    # Calcular número de muestras por categoría
    samples_per_category = {}
    remaining = num_samples

    for cat, info in categories.items():
        if cat == list(categories.keys())[-1]:
            # La última categoría toma todas las muestras restantes
            samples_per_category[cat] = remaining
        else:
            # Calcular según proporción
            cat_samples = int(num_samples * info["proportion"])
            samples_per_category[cat] = cat_samples
            remaining -= cat_samples

    # Crear directorios de salida
    os.makedirs(output_dir, exist_ok=True)

    for cat in categories.keys():
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    # Generar muestras
    stats = {cat: 0 for cat in categories.keys()}

    progress_bar = tqdm(total=num_samples, desc="Generando modelos")

    for cat, num_cat_samples in samples_per_category.items():
        generator = categories[cat]["generator"]

        for i in range(num_cat_samples):
            try:
                # Generar modelo
                mesh = generator()

                # Voxelizar
                voxels = voxelize_mesh(mesh, resolution=resolution)

                # Guardar como numpy array
                sample_path = os.path.join(output_dir, cat, f"{cat}_{i + 1:04d}")
                np.save(f"{sample_path}.npy", voxels)

                # Guardar modelo como STL
                mesh.export(f"{sample_path}.stl")

                # Visualizar si se solicita
                if visualize and i % 10 == 0:  # Visualizar cada 10 muestras
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection='3d')

                    # Visualizar voxels
                    voxels_binary = voxels > 0.5
                    x, y, z = np.indices((resolution, resolution, resolution))
                    ax.voxels(voxels_binary, edgecolor='k', alpha=0.2)

                    # Configurar visualización
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f"{cat} {i + 1}")

                    # Guardar visualización
                    plt.savefig(f"{sample_path}_preview.png", bbox_inches='tight')
                    plt.close()

                stats[cat] += 1
                progress_bar.update(1)

            except Exception as e:
                print(f"Error generando {cat} {i + 1}: {str(e)}")

    progress_bar.close()

    # Guardar estadísticas
    with open(os.path.join(output_dir, "stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Generación completada. Total: {sum(stats.values())} modelos")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generar conjunto de datos sintético de modelos 3D decorativos")

    parser.add_argument('--num_samples', type=int, default=100, help='Número de muestras a generar')
    parser.add_argument('--output_dir', type=str, default='data/synthetic', help='Directorio de salida')
    parser.add_argument('--resolution', type=int, default=32, help='Resolución de voxelización')
    parser.add_argument('--visualize', action='store_true', help='Visualizar modelos generados')

    # Proporciones de categorías
    parser.add_argument('--vase_proportion', type=float, default=0.3, help='Proporción de floreros')
    parser.add_argument('--planter_proportion', type=float, default=0.2, help='Proporción de macetas')
    parser.add_argument('--bowl_proportion', type=float, default=0.2, help='Proporción de cuencos')
    parser.add_argument('--candleholder_proportion', type=float, default=0.15, help='Proporción de portavelas')
    parser.add_argument('--plate_proportion', type=float, default=0.15, help='Proporción de platos')

    args = parser.parse_args()

    # Configurar categorías
    categories = {
        "vase": {"generator": generate_random_vase, "proportion": args.vase_proportion},
        "planter": {"generator": generate_random_planter, "proportion": args.planter_proportion},
        "bowl": {"generator": generate_random_bowl, "proportion": args.bowl_proportion},
        "candleholder": {"generator": generate_random_candleholder, "proportion": args.candleholder_proportion},
        "plate": {"generator": generate_random_plate, "proportion": args.plate_proportion}
    }

    # Generar conjunto de datos
    create_synthetic_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        categories=categories,
        resolution=args.resolution,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()