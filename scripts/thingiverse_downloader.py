# thingiverse_downloader.py
import os
import requests
import json
import time
import zipfile
from tqdm import tqdm


def download_from_thingiverse(api_key, category="home", keyword="decor", min_likes=10, max_models=50,
                              output_dir="data/thingiverse"):
    """
    Descarga modelos 3D de Thingiverse usando su API oficial.

    Args:
        api_key: Clave API de Thingiverse
        category: Categoría de modelos a buscar
        keyword: Palabra clave para la búsqueda
        min_likes: Número mínimo de likes para filtrar
        max_models: Número máximo de modelos a descargar
        output_dir: Directorio de salida

    Returns:
        int: Número de modelos descargados
    """
    os.makedirs(output_dir, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "DecoraAI/1.0 (research-project)"
    }

    # Crear registro de metadatos
    metadata_file = os.path.join(output_dir, "metadata.json")
    metadata = []

    # Buscar modelos
    page = 1
    total_downloaded = 0

    while total_downloaded < max_models:
        print(f"Procesando página {page}")

        url = f"https://api.thingiverse.com/search/{keyword}"
        params = {
            "page": page,
            "per_page": 20,
            "sort": "popularity",
            "category": category
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            hits = data.get("hits", [])

            if not hits:
                print("No hay más resultados")
                break

            for item in hits:
                if total_downloaded >= max_models:
                    break

                likes = item.get("like_count", 0)
                if likes < min_likes:
                    continue

                thing_id = item.get("id")
                name = item.get("name", "").replace(" ", "_")

                print(f"Descargando {name} (ID: {thing_id}, Likes: {likes})")

                # Obtener información del modelo
                thing_url = f"https://api.thingiverse.com/things/{thing_id}"
                thing_response = requests.get(thing_url, headers=headers)
                thing_response.raise_for_status()
                thing_data = thing_response.json()

                # Obtener archivos
                files_url = f"https://api.thingiverse.com/things/{thing_id}/files"
                files_response = requests.get(files_url, headers=headers)
                files_response.raise_for_status()
                files_data = files_response.json()

                # Descargar archivos 3D
                model_dir = os.path.join(output_dir, f"{thing_id}_{name[:30]}")
                os.makedirs(model_dir, exist_ok=True)

                files_downloaded = []
                has_3d_files = False

                for file_data in files_data:
                    file_name = file_data.get("name", "")
                    download_url = file_data.get("download_url")

                    if not download_url:
                        continue

                    # Verificar si es un archivo 3D o ZIP
                    is_3d = any(file_name.lower().endswith(ext) for ext in ['.stl', '.obj', '.ply'])
                    is_zip = file_name.lower().endswith('.zip')

                    if is_3d or is_zip:
                        file_path = os.path.join(model_dir, file_name)

                        file_response = requests.get(download_url, headers=headers, stream=True)
                        file_response.raise_for_status()

                        total_size = int(file_response.headers.get('content-length', 0))

                        with open(file_path, 'wb') as f, tqdm(
                                desc=file_name,
                                total=total_size,
                                unit='B',
                                unit_scale=True,
                                unit_divisor=1024,
                        ) as bar:
                            for chunk in file_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    bar.update(len(chunk))

                        files_downloaded.append(file_name)

                        # Extraer ZIP si es necesario
                        if is_zip:
                            try:
                                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                    extracted_files = [f for f in zip_ref.namelist()
                                                       if f.lower().endswith(('.stl', '.obj', '.ply'))]
                                    if extracted_files:
                                        zip_ref.extractall(model_dir)
                                        has_3d_files = True

                                    # Opcional: eliminar ZIP después de extraer
                                    # os.remove(file_path)
                            except Exception as e:
                                print(f"Error extrayendo ZIP: {e}")
                        else:
                            has_3d_files = True

                if not has_3d_files and not files_downloaded:
                    print(f"No se encontraron archivos 3D para {name}")
                    continue

                # Guardar metadatos
                metadata.append({
                    "id": thing_id,
                    "name": name,
                    "source": "thingiverse",
                    "license": thing_data.get("license"),
                    "likes": likes,
                    "files": files_downloaded,
                    "url": thing_data.get("public_url", ""),
                    "creator": thing_data.get("creator", {}).get("name", "unknown"),
                    "download_date": time.strftime("%Y-%m-%d")
                })

                total_downloaded += 1
                print(f"Descargado modelo {total_downloaded}/{max_models}")

            # Guardar metadatos actualizados
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Pasar a la siguiente página
            page += 1

            # Esperar para no sobrecargar la API
            time.sleep(2)

        except Exception as e:
            print(f"Error procesando página {page}: {e}")
            time.sleep(5)  # Esperar más tiempo si hay un error

    print(f"Descarga completa: {total_downloaded} modelos en {output_dir}")
    return total_downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descargar modelos 3D de Thingiverse")
    parser.add_argument('--api_key', type=str, required=True, help='Clave API de Thingiverse')
    parser.add_argument('--category', type=str, default="home", help='Categoría de modelos')
    parser.add_argument('--keyword', type=str, default="decor", help='Palabra clave para búsqueda')
    parser.add_argument('--min_likes', type=int, default=10, help='Número mínimo de likes')
    parser.add_argument('--max_models', type=int, default=50, help='Número máximo de modelos')
    parser.add_argument('--output_dir', type=str, default="data/thingiverse", help='Directorio de salida')

    args = parser.parse_args()

    download_from_thingiverse(
        api_key=args.api_key,
        category=args.category,
        keyword=args.keyword,
        min_likes=args.min_likes,
        max_models=args.max_models,
        output_dir=args.output_dir
    )