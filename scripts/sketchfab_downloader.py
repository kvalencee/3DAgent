# sketchfab_downloader.py
import os
import requests
import json
import time
import zipfile
from tqdm import tqdm


def download_from_sketchfab(api_key, category=None, keyword=None, count=20, output_dir="data/sketchfab"):
    """
    Descarga modelos 3D gratuitos de Sketchfab que tengan licencias permisivas.

    Args:
        api_key: Clave API de Sketchfab
        category: Categoría de modelos a buscar
        count: Número de modelos a descargar
        output_dir: Directorio de salida

    Returns:
        int: Número de modelos descargados
    """
    os.makedirs(output_dir, exist_ok=True)

    headers = {
        "Authorization": f"Token {api_key}"
    }

    # Solo descargar modelos con licencias que permitan redistribución
    allowed_licenses = [
        "CC0", "CC-BY", "CC-BY-SA", "CC-BY-ND", "CC-BY-NC", "CC-BY-NC-SA", "CC-BY-NC-ND"
    ]

    # Filtrar solo modelos gratuitos con formato descargable
    url = "https://api.sketchfab.com/v3/models"
    params = {
        "downloadable": True,
        "count": 100,  # Solicitar más para poder filtrar
        "q": keyword  # Añadir parámetro de búsqueda
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        models = data.get("results", [])

        # Filtrar modelos por licencia
        legal_models = [m for m in models if m.get("license", {}).get("slug") in allowed_licenses]

        print(f"Encontrados {len(legal_models)} modelos con licencias permitidas")

        # Metadatos
        metadata_file = os.path.join(output_dir, "metadata.json")
        metadata = []

        # Descargar hasta 'count' modelos
        downloaded = 0

        for model in legal_models[:count]:
            model_uid = model.get("uid")
            model_name = model.get("name", "").replace(" ", "_")
            license_info = model.get("license", {}).get("slug", "unknown")

            print(f"Procesando {model_name} (UID: {model_uid}, Licencia: {license_info})")

            # Obtener enlace de descarga
            download_url = f"https://api.sketchfab.com/v3/models/{model_uid}/download"
            dl_response = requests.get(download_url, headers=headers)
            dl_response.raise_for_status()

            dl_data = dl_response.json()
            archive_url = dl_data.get("gltf", {}).get("url")

            if not archive_url:
                print("No se encontró URL de descarga")
                continue

            # Crear directorio para el modelo
            model_dir = os.path.join(output_dir, f"{model_uid}_{model_name[:30]}")
            os.makedirs(model_dir, exist_ok=True)

            # Descargar archivo
            archive_path = os.path.join(model_dir, "model.zip")

            dl_file_response = requests.get(archive_url, stream=True)
            dl_file_response.raise_for_status()

            total_size = int(dl_file_response.headers.get('content-length', 0))

            with open(archive_path, 'wb') as f, tqdm(
                    desc="Descargando",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in dl_file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            # Extraer archivo ZIP
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    extracted_files = zip_ref.namelist()
                    zip_ref.extractall(model_dir)

                # Opcional: eliminar ZIP después de extraer
                # os.remove(archive_path)

                # Guardar metadatos
                metadata.append({
                    "id": model_uid,
                    "name": model_name,
                    "source": "sketchfab",
                    "license": license_info,
                    "files": extracted_files,
                    "url": model.get("viewerUrl", ""),
                    "creator": model.get("user", {}).get("username", "unknown"),
                    "download_date": time.strftime("%Y-%m-%d")
                })

                downloaded += 1
                print(f"Descargado modelo {downloaded}/{count}")

            except Exception as e:
                print(f"Error extrayendo ZIP: {e}")

        # Guardar metadatos
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Descarga completa: {downloaded} modelos en {output_dir}")
        return downloaded

    except Exception as e:
        print(f"Error descargando modelos: {e}")
        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descargar modelos 3D de Sketchfab")
    parser.add_argument('--api_key', type=str, required=True, help='Clave API de Sketchfab')
    parser.add_argument('--category', type=str, default="furniture", help='Categoría de modelos')
    parser.add_argument('--count', type=int, default=20, help='Número de modelos a descargar')
    parser.add_argument('--output_dir', type=str, default="data/sketchfab", help='Directorio de salida')
    parser.add_argument('--keyword', type=str, default=None, help='Palabra clave para búsqueda')

    args = parser.parse_args()

    download_from_sketchfab(
        api_key=args.api_key,
        category=args.category,
        count=args.count,
        output_dir=args.output_dir
    )