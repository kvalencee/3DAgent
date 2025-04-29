# academic_downloader.py
import os
import requests
import zipfile
import io
import json
import time
from tqdm import tqdm


def download_academic_dataset(name="ModelNet10", category=None, output_dir="data/academic"):
    """
    Descarga conjuntos de datos académicos comunes.

    Args:
        name: Nombre del dataset a descargar
        category: Categoría específica a extraer (opcional)
        output_dir: Directorio de salida

    Returns:
        int: Número de archivos extraídos
    """
    os.makedirs(output_dir, exist_ok=True)

    # URLs de los datasets (usa URLs públicas siempre que sea posible)
    datasets = {
        "ModelNet10": {
            "url": "http://3dshapenets.cs.princeton.edu/ModelNet10.zip",
            "categories": ["chair", "table", "sofa", "bed", "toilet", "desk", "bathtub", "night_stand", "dresser",
                           "monitor"]
        },
        "ModelNet40": {
            "url": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
            "categories": ["vase", "lamp", "bowl", "flower_pot", "bottle", "cup", "plant", "radio", "stool", "wardrobe"]
        }
    }

    if name not in datasets:
        print(f"Dataset '{name}' no soportado. Opciones disponibles: {list(datasets.keys())}")
        return 0

    dataset_info = datasets[name]
    url = dataset_info["url"]
    categories_list = dataset_info["categories"]

    # Filtrar por categoría si se especifica
    if category and category in categories_list:
        categories_list = [category]
    elif category and category not in categories_list:
        print(f"Categoría '{category}' no encontrada en {name}. Categorías disponibles: {categories_list}")
        categories_list = []  # No descargar nada si la categoría es inválida

    print(f"Descargando dataset {name} desde {url}")
    print(f"Categorías a extraer: {categories_list}")

    try:
        # Descargar archivo ZIP
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Mostrar progreso de descarga
        total_size = int(response.headers.get('content-length', 0))

        # Usar BytesIO para no guardar el ZIP completo en disco
        with io.BytesIO() as file_stream:
            with tqdm(
                    desc=f"Descargando {name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_stream.write(chunk)
                        bar.update(len(chunk))

            # Volver al inicio del stream
            file_stream.seek(0)

            # Extraer solo las categorías seleccionadas
            extracted_files = 0

            with zipfile.ZipFile(file_stream) as zip_ref:
                # Listar todos los archivos
                file_list = zip_ref.namelist()

                # Filtrar por categorías
                files_to_extract = []
                for category_name in categories_list:
                    category_files = [f for f in file_list if f'/{category_name}/' in f.lower()]
                    files_to_extract.extend(category_files)

                # Extraer archivos
                for file in tqdm(files_to_extract, desc="Extrayendo archivos"):
                    zip_ref.extract(file, output_dir)
                    extracted_files += 1

        # Guardar metadatos
        metadata_file = os.path.join(output_dir, f"{name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump({
                "name": name,
                "source": url,
                "categories": categories_list,
                "download_date": time.strftime("%Y-%m-%d"),
                "files_count": extracted_files
            }, f, indent=2)

        print(f"Dataset {name} descargado y extraído en {output_dir}")
        print(f"Se extrajeron {extracted_files} archivos")
        return extracted_files

    except Exception as e:
        print(f"Error descargando dataset {name}: {e}")
        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descargar datasets académicos de modelos 3D")
    parser.add_argument('--dataset', type=str, default="ModelNet10",
                        choices=["ModelNet10", "ModelNet40"],
                        help='Dataset a descargar')
    parser.add_argument('--category', type=str, default=None,
                        help='Categoría específica a extraer (opcional)')
    parser.add_argument('--output_dir', type=str, default="data/academic",
                        help='Directorio de salida')

    args = parser.parse_args()

    download_academic_dataset(
        name=args.dataset,
        category=args.category,
        output_dir=args.output_dir
    )