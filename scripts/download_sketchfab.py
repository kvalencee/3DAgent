import os
import requests
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración
API_KEY = "99d4a425f4be42a9aa29883e98ccbf17"
OUTPUT_DIR = r"C:\Users\Kevin Valencia\Documents\ESCOM\DECORAI\data\vases_ai"
MAX_MODELS = 1000
MIN_FACES = 1500  # Filtro de calidad
THREADS = 3  # Hilos paralelos


def sanitize_name(name):
    """Elimina caracteres inválidos en nombres de archivo"""
    return "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()


def fetch_models(keyword, offset=0):
    """Obtiene modelos paginados de Sketchfab"""
    url = "https://api.sketchfab.com/v3/models"
    params = {
        "downloadable": True,
        "count": 100,  # Máximo permitido por petición
        "offset": offset,
        "q": keyword,
        "sort_by": "-likeCount",  # Modelos populares = mejor calidad
        "min_face_count": MIN_FACES,
        "type": "models"
    }
    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Error al buscar modelos: {e}")
        return []


def download_model(model):
    """Descarga un modelo en formato .glb"""
    try:
        model_id = model["uid"]
        model_name = sanitize_name(model["name"])
        license_type = model.get("license", {}).get("slug", "unknown")

        # Verificar si ya existe
        output_path = os.path.join(OUTPUT_DIR, f"{model_id}_{model_name}.glb")
        if os.path.exists(output_path):
            return None

        # Obtener URL de descarga GLB
        dl_url = f"https://api.sketchfab.com/v3/models/{model_id}/download"
        headers = {"Authorization": f"Token {API_KEY}"}
        response = requests.get(dl_url, headers=headers, timeout=30)
        response.raise_for_status()
        glb_url = response.json().get("glb", {}).get("url")

        if not glb_url:
            return None

        # Descargar archivo
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with requests.get(glb_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f, tqdm(
                    unit='B', unit_scale=True,
                    desc=f"Descargando {model_name[:15]}...",
                    total=int(r.headers.get('content-length', 0))
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        # Metadatos para IA
        return {
            "id": model_id,
            "file": f"{model_id}_{model_name}.glb",
            "name": model_name,
            "license": license_type,
            "faces": model.get("faceCount"),
            "vertices": model.get("vertexCount"),
            "bounding_box": model.get("boundingBox", {}).get("dimensions")
        }

    except Exception as e:
        print(f"Error en {model.get('uid')}: {str(e)[:100]}")
        return None


def main():
    keywords = [
        "vase -ancient -greek",  # Excluir jarrones antiguos
        "modern vase",
        "ceramic vase",
        "3d scan vase"
    ]

    metadata = []
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = []
        for keyword in keywords:
            models = fetch_models(keyword)
            for model in models[:MAX_MODELS // len(keywords)]:
                futures.append(executor.submit(download_model, model))
                time.sleep(1.5)  # Respeta rate limits

        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando"):
            if result := future.result():
                metadata.append(result)

    # Guardar metadatos
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Descargados {len(metadata)} modelos en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()