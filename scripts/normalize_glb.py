import bpy
import os
import sys


def normalize_glb(input_path, output_path):
    """Normaliza un modelo GLB en Blender"""
    # Borrar escena inicial
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Importar GLB
    bpy.ops.import_scene.gltf(filepath=input_path)

    # Seleccionar todos los objetos
    bpy.ops.object.select_all(action='SELECT')

    # Centrar geometría
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.location_clear()

    # Escalar a 1 metro de altura
    max_dim = max(bpy.context.object.dimensions)
    scale_factor = 1.0 / max_dim
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))

    # Exportar
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_yup=True,  # Eje Y arriba (estándar)
    )


if __name__ == "__main__":
    input_path = sys.argv[-2]
    output_path = sys.argv[-1]
    normalize_glb(input_path, output_path)