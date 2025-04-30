import os
import requests
import json
import time
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from urllib.parse import quote
import backoff  # pip install backoff

API_KEY = "99d4a425f4be42a9aa29883e98ccbf17"
OUTPUT_DIR = r"C:\Users\Kevin Valencia\Documents\ESCOM\DECORAI\data\sketchfab\vases"
MAX_MODELS = 1000  # Total deseado
THREADS = 5  # Descargas paralelas


def fetch_models(keyword, offset=0, count=100):
    """Obtiene modelos paginados desde la API."""
    url = "https://api.sketchfab.com/v3/models"
    params = {
        "downloadable": True,
        "count": count,
        "offset": offset,
        "q": keyword,
        "sort_by": "-likeCount",  # Los mejores primero
        "license": "cc0,cc-by,cc-by-nc"  # Filtro de licencias
    }
    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def download_model(model):
    """Descarga y extrae un modelo individual."""
    model_uid = model["uid"]
    model_name = model["name"].replace(" ", "_")[:50]
    model_dir = os.path.join(OUTPUT_DIR, f"{model_uid}_{model_name}")
    os.makedirs(model_dir, exist_ok=True)

    # Obtener URL de descarga
    try:
        dl_url = f"https://api.sketchfab.com/v3/models/{model_uid}/download"
        dl_response = requests.get(dl_url, headers={"Authorization": f"Token {API_KEY}"})
        dl_response.raise_for_status()
        archive_url = dl_response.json().get("gltf", {}).get("url")
        if not archive_url:
            return None
    except Exception as e:
        print(f"Error getting download URL for {model_uid}: {e}")
        return None

    # Descargar archivo
    archive_path = os.path.join(model_dir, "model.zip")
    try:
        with requests.get(archive_url, stream=True) as r:
            r.raise_for_status()
            with open(archive_path, 'wb') as f, tqdm(
                    unit='B', unit_scale=True, desc=f"Descargando {model_name}"
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        # Extraer y limpiar
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(archive_path)  # Eliminar ZIP después de extraer

        return {
            "id": model_uid,
            "name": model["name"],
            "license": model["license"]["slug"],
            "files": os.listdir(model_dir),
            "url": model["viewerUrl"]
        }
    except Exception as e:
        print(f"Error downloading {model_uid}: {e}")
        return None


def sanitize_filename(name):
    """Elimina caracteres inválidos para nombres de archivo en Windows"""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)  # Reemplaza caracteres prohibidos
    name = name[:100]  # Limita la longitud
    return name.strip()


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def safe_api_request(url, headers, params=None):
    """Maneja reintentos automáticos para errores 429"""
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response


def download_model(model):
    try:
        model_name = sanitize_filename(model["name"])
        model_uid = model["uid"]
        model_dir = os.path.join(OUTPUT_DIR, f"{model_uid}_{model_name}")

        # Verificar si ya existe
        if os.path.exists(model_dir):
            print(f"Modelo {model_uid} ya descargado, omitiendo...")
            return None

        # Obtener URL de descarga con manejo de rate limiting
        dl_url = f"https://api.sketchfab.com/v3/models/{model_uid}/download"
        dl_response = safe_api_request(dl_url, headers={"Authorization": f"Token {API_KEY}"})
        archive_url = dl_response.json().get("gltf", {}).get("url")

        # Descargar archivo
        os.makedirs(model_dir, exist_ok=True)
        archive_path = os.path.join(model_dir, "model.zip")

        with requests.get(archive_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(archive_path, 'wb') as f, tqdm(
                    unit='B', unit_scale=True, desc=f"Descargando {model_name[:20]}...",
                    total=int(r.headers.get('content-length', 0))
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        # Extraer y limpiar
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(archive_path)

        return {
            "id": model_uid,
            "name": model_name,
            "license": model.get("license", {}).get("slug", "unknown"),
            "files": os.listdir(model_dir),
            "url": model.get("viewerUrl", "")
        }

    except Exception as e:
        print(f"Error procesando modelo {model.get('uid')}: {str(e)[:200]}")
        return None


def main():
    # Configuración inicial
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = []
    keywords = ["vase", "jarrón", "ceramic vase", "flower pot", "urn vase"]

    # Control de velocidad (1 petición/segundo)
    with ThreadPoolExecutor(max_workers=3) as executor:  # Reducir workers
        futures = []
        for keyword in keywords:
            models = fetch_models(keyword, count=100)
            for model in models[:MAX_MODELS // len(keywords)]:
                futures.append(executor.submit(download_model, model))
                time.sleep(1.5)  # Espacio entre peticiones

        for future in tqdm(as_completed(futures), total=len(futures), desc="Descargando"):
            if result := future.result():
                metadata.append(result)

    # Guardar metadatos
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()