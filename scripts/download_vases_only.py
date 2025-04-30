import os
import requests
import time
from tqdm import tqdm

# Configuración
API_KEY = "99d4a425f4be42a9aa29883e98ccbf17"
OUTPUT_DIR = r"C:\Users\Kevin Valencia\Documents\ESCOM\DECORAI\data\vases_ai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de búsqueda
SEARCH_QUERIES = [
    "vase",
    "jarrón",
    "flower pot",
    "ceramic vase",
    "porcelain vase"
]

# Filtros avanzados
PARAMS = {
    "downloadable": "true",
    "staffpicked": "true",  # Solo modelos verificados
    "min_face_count": "1000",  # Modelos detallados
    "max_face_count": "20000",  # Evitar modelos demasiado complejos
    "sort_by": "-likeCount",  # Los más populares primero
    "count": "100"  # Máximo permitido por petición
}


def download_model(model_id):
    """Descarga un modelo individual en formato .glb"""
    try:
        # Paso 1: Obtener URL de descarga
        dl_url = f"https://api.sketchfab.com/v3/models/{model_id}/download"
        headers = {"Authorization": f"Token {API_KEY}"}
        response = requests.get(dl_url, headers=headers, timeout=30)
        response.raise_for_status()

        glb_url = response.json().get("glb", {}).get("url")
        if not glb_url:
            return False

        # Paso 2: Descargar archivo
        filename = f"vase_{model_id}.glb"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with requests.get(glb_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True

    except Exception as e:
        print(f"Error al descargar {model_id}: {str(e)[:100]}")
        return False


def fetch_vase_models():
    """Busca modelos de jarrones con filtros estrictos"""
    headers = {"Authorization": f"Token {API_KEY}"}
    downloaded = 0
    seen_models = set()  # Para evitar duplicados

    for query in SEARCH_QUERIES:
        print(f"\n🔍 Buscando: '{query}'")
        try:
            params = PARAMS.copy()
            params["q"] = query

            response = requests.get(
                "https://api.sketchfab.com/v3/models",
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            models = response.json().get("results", [])

            for model in tqdm(models, desc="Procesando resultados"):
                model_id = model["uid"]

                # Verificar si ya lo hemos procesado
                if model_id in seen_models:
                    continue
                seen_models.add(model_id)

                # Descargar el modelo
                if download_model(model_id):
                    downloaded += 1
                    print(f"✅ Descargado: vase_{model_id}.glb")

                time.sleep(1.5)  # Respetar rate limits

        except Exception as e:
            print(f"Error en búsqueda '{query}': {str(e)[:100]}")

    return downloaded


if __name__ == "__main__":
    print("🚀 Iniciando descarga de jarrones...")
    total = fetch_vase_models()
    print(f"\n🎉 Descarga completada! {total} jarrones guardados en:")
    print(OUTPUT_DIR)