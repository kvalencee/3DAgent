import os
import requests
import json
import time
from tqdm import tqdm
import backoff

# Configuración
API_KEY = "TU_API_KEY"  # Reemplaza con tu API key
OUTPUT_DIR = "data/vases"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Términos de búsqueda específicos para floreros
SEARCH_TERMS = [
    "vase flower",
    "ceramic vase",
    "decorative vase",
    "modern vase",
    "3d printable vase",
    "flower pot",
    "planter vase"
]

# Filtros para asegurar calidad
PARAMS = {
    "downloadable": "true",
    "min_face_count": "2000",  # Asegurar suficiente detalle
    "max_face_count": "50000",  # Evitar modelos demasiado complejos
    "sort_by": "-likeCount",  # Preferir modelos populares
    "count": "24",  # Resultados por página
    "license": "cc0,cc-by"  # Solo licencias permisivas
}


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def get_models(search_term, offset=0):
    """Obtiene modelos de la API de Sketchfab con reintento automático"""
    url = "https://api.sketchfab.com/v3/models"
    params = PARAMS.copy()
    params["q"] = search_term
    params["offset"] = offset

    headers = {"Authorization": f"Token {API_KEY}"}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def download_vase(model_id, model_name):
    """Descarga un modelo en formato GLB"""
    output_path = os.path.join(OUTPUT_DIR, f"{model_id}_{model_name}.glb")

    # Verificar si ya existe
    if os.path.exists(output_path):
        return output_path

    # Obtener URL de descarga
    dl_url = f"https://api.sketchfab.com/v3/models/{model_id}/download"
    headers = {"Authorization": f"Token {API_KEY}"}
    response = requests.get(dl_url, headers=headers)
    glb_url = response.json().get("glb", {}).get("url")

    if not glb_url:
        print(f"No GLB disponible para {model_name}")
        return None

    # Descargar el archivo
    print(f"Descargando: {model_name}")
    with requests.get(glb_url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f, tqdm(
                total=int(r.headers.get('content-length', 0)),
                unit='B', unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    return output_path


def collect_vases(max_models=50):
    """Recolecta floreros desde Sketchfab"""
    metadata = []
    total_downloaded = 0

    for term in SEARCH_TERMS:
        print(f"\nBuscando: '{term}'")
        offset = 0

        while total_downloaded < max_models:
            try:
                result = get_models(term, offset)
                models = result.get("results", [])

                if not models:
                    print(f"No más resultados para '{term}'")
                    break

                for model in models:
                    model_id = model["uid"]
                    model_name = model["name"].replace(" ", "_")[:50]

                    # Verificar que sea apropiado para impresión 3D
                    is_printable = any(tag.lower() in ["printable", "3d print", "3d printing"]
                                       for tag in model.get("tags", []))

                    # Descargar el modelo
                    file_path = download_vase(model_id, model_name)

                    if file_path:
                        metadata.append({
                            "id": model_id,
                            "name": model["name"],
                            "file": os.path.basename(file_path),
                            "printable": is_printable,
                            "faces": model.get("faceCount"),
                            "vertices": model.get("vertexCount"),
                            "url": model.get("viewerUrl")
                        })
                        total_downloaded += 1

                        if total_downloaded >= max_models:
                            break

                # Pasar a la siguiente página
                offset += len(models)
                time.sleep(1.5)  # Respetar rate limits

            except Exception as e:
                print(f"Error procesando '{term}': {str(e)}")
                break

    # Guardar metadata
    with open(os.path.join(OUTPUT_DIR, "vases_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCompletado. {total_downloaded} floreros descargados.")
    return metadata


if __name__ == "__main__":
    collect_vases(max_models=100)