import os
import requests
import zipfile
import io
import time
from tqdm import tqdm

# Configuración
OUTPUT_DIR = "dataset/raw/modelnet"
METADATA_FILE = "dataset/metadata/modelnet_metadata.txt"

# Crear directorios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)

# URLs de datasets
datasets = [
    {
        "name": "ModelNet10",
        "url": "http://3dshapenets.cs.princeton.edu/ModelNet10.zip",
        "categories": ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table",
                       "toilet"]
    },
    {
        "name": "ModelNet40",
        "url": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
        "categories": ["vase", "lamp", "flower_pot", "bowl"]  # Solo categorías relevantes para decoración
    }
]


def download_dataset(dataset):
    """Descarga y extrae un dataset académico."""
    name = dataset["name"]
    url = dataset["url"]
    output_path = os.path.join(OUTPUT_DIR, name)

    print(f"Descargando {name}...")

    try:
        # Descargar archivo
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Preparar para mostrar progreso
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        # Descargar con barra de progreso
        with io.BytesIO() as file_stream:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=name) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file_stream.write(chunk)
                        pbar.update(len(chunk))

            # Volver al inicio del stream
            file_stream.seek(0)

            # Extraer archivo
            print(f"Extrayendo {name}...")
            with zipfile.ZipFile(file_stream) as zip_ref:
                # Extraer solo categorías relevantes
                for item in zip_ref.infolist():
                    category_match = False
                    for category in dataset["categories"]:
                        if f"/{category}/" in item.filename:
                            category_match = True
                            break

                    if category_match or len(dataset["categories"]) == 0:
                        zip_ref.extract(item, output_path)

        print(f"{name} descargado y extraído correctamente")
        return True

    except Exception as e:
        print(f"Error descargando {name}: {e}")
        return False


def main():
    print("Descargando datasets académicos para entrenamiento...")

    with open(METADATA_FILE, 'w') as f:
        f.write(f"Datasets descargados el {time.strftime('%Y-%m-%d')}:\n\n")

        for dataset in datasets:
            success = download_dataset(dataset)

            if success:
                f.write(f"- {dataset['name']}\n")
                f.write(f"  URL: {dataset['url']}\n")
                f.write(f"  Categorías: {', '.join(dataset['categories'])}\n\n")

    print(f"Proceso completado. Datasets guardados en {OUTPUT_DIR}")
    print(f"Metadata guardada en {METADATA_FILE}")


if __name__ == "__main__":
    main()