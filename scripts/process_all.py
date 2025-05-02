# process_all.py
import os
import argparse
import subprocess
import time
import datetime


def run_command(command, description):
    """Ejecuta un comando con descripción"""
    print(f"\n{'=' * 50}")
    print(f"EJECUTANDO: {description}")
    print(f"COMANDO: {command}")
    print(f"{'=' * 50}\n")

    start_time = time.time()
    result = subprocess.run(command, shell=True)
    end_time = time.time()

    print(f"\n{'=' * 50}")
    print(f"COMPLETADO: {description}")
    print(f"TIEMPO: {datetime.timedelta(seconds=int(end_time - start_time))}")
    print(f"CÓDIGO DE SALIDA: {result.returncode}")
    print(f"{'=' * 50}\n")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Procesa modelos 3D de floreros desde STL hasta entrenamiento")
    parser.add_argument('--skip_normalization', action='store_true', help='Omitir normalización')
    parser.add_argument('--skip_labeling', action='store_true', help='Omitir etiquetado')
    parser.add_argument('--skip_voxelization', action='store_true', help='Omitir voxelización')
    parser.add_argument('--skip_training', action='store_true', help='Omitir preparación para entrenamiento')
    parser.add_argument('--resolution', type=int, default=64, help='Resolución de voxelización')

    args = parser.parse_args()

    # Rutas base
    base_dir = r"C:\Users\Kevin Valencia\Documents\ESCOM\DECORAI"
    raw_dir = os.path.join(base_dir, "data", "raw")
    normalized_dir = os.path.join(base_dir, "data", "normalized")
    voxels_dir = os.path.join(base_dir, "data", f"voxels_{args.resolution}")
    training_dir = os.path.join(base_dir, "data", "training_ready")
    metadata_dir = os.path.join(base_dir, "metadata")
    metadata_file = os.path.join(metadata_dir, "vase_metadata.csv")
    printability_file = os.path.join(metadata_dir, "printability_analysis.csv")

    # Crear directorios necesarios
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(voxels_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Paso 1: Normalización
    if not args.skip_normalization:
        success = run_command(
            f'python batch_process.py --input_dir "{raw_dir}" --output_dir "{normalized_dir}"',
            "Normalización de modelos"
        )

        if not success:
            print("Error en normalización. Abortando proceso.")
            return

        # Analizar imprimibilidad
        run_command(
            f'python printability_checker.py --input_dir "{normalized_dir}" --output_file "{printability_file}"',
            "Análisis de imprimibilidad"
        )

    # Paso 2: Etiquetado
    if not args.skip_labeling:
        print("\n" + "=" * 50)
        print("ETIQUETADO MANUAL")
        print("Ejecute el siguiente comando en una terminal separada:")
        print(f'python vase_labeling_tool.py "{normalized_dir}" "{metadata_file}"')
        print("Cuando termine el etiquetado, continúe este script.")
        print("=" * 50 + "\n")

        input("Presione Enter cuando haya completado el etiquetado...")

    # Paso 3: Voxelización
    if not args.skip_voxelization:
        success = run_command(
            f'python voxelizer.py --input_dir "{normalized_dir}" --output_dir "{voxels_dir}" --resolution {args.resolution}',
            f"Voxelización a resolución {args.resolution}"
        )

        if not success:
            print("Error en voxelización. Abortando proceso.")
            return

    # Paso 4: Preparación para entrenamiento
    if not args.skip_training:
        success = run_command(
            f'python prepare_training_data.py --voxel_dir "{voxels_dir}" --metadata_file "{metadata_file}" --output_dir "{training_dir}" --balanced',
            "Preparación de datos para entrenamiento"
        )

        if not success:
            print("Error en preparación para entrenamiento. Abortando proceso.")
            return

    print("\n" + "=" * 50)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print(f"Modelos normalizados: {normalized_dir}")
    print(f"Voxels: {voxels_dir}")
    print(f"Datos de entrenamiento: {training_dir}")
    print(f"Metadatos: {metadata_file}")
    print("=" * 50 + "\n")

    print("Para entrenar los modelos, ejecute los siguientes comandos:")
    train_dir = os.path.join(training_dir, "train")
    models_dir = os.path.join(base_dir, "models")
    print(
        f'python scripts/train_vae.py --data_dir "{train_dir}" --output_dir "{models_dir}" --resolution {args.resolution} --latent_dim 128 --epochs 200')
    print(
        f'python scripts/train_gan.py --data_dir "{train_dir}" --output_dir "{models_dir}" --resolution {args.resolution} --conditional')


if __name__ == "__main__":
    main()