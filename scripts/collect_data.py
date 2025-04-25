import os
import sys
import argparse
import requests
import time
import json
import random
from urllib.parse import urlparse
from tqdm import tqdm
import zipfile
import shutil

# Añadir la carpeta raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Datos de API (para Thingiverse necesitas un token)
THINGIVERSE_API_BASE = "https://api.thingiverse.com"
DEFAULT_USER_AGENT = "DecoraAI/1.0 (https://github.com/yourname/decorai)"


def download_file(url, destination, session=None, retry=3, timeout=30, chunk_size=8192):
    """
    Descarga un archivo desde una URL.

    Args:
        url: URL del archivo
        destination: Ruta de destino
        session: Sesión de requests (opcional)
        retry: Número de reintentos
        timeout: Tiempo de espera en segundos
        chunk_size: Tamaño de bloque para descarga

    Returns:
        bool: True si se descargó correctamente, False si falló
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Usar sesión existente o crear una nueva
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    # Intentar descarga con reintentos
    for attempt in range(retry):
        try:
            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Calcular tamaño total y mostrar barra de progreso
            total_size = int(response.headers.get('content-length', 0))

            with open(destination, 'wb') as file, tqdm(
                    desc=os.path.basename(destination),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filtrar keep-alive new chunks
                        file.write(chunk)
                        progress.update(len(chunk))

            return True

        except (requests.RequestException, IOError) as e:
            print(f"Intento {attempt + 1}/{retry} falló: {str(e)}")
            time.sleep(2)  # Esperar antes de reintentar

    print(f"Error: No se pudo descargar {url} después de {retry} intentos")
    return False


def extract_zip(zip_path, extract_path, remove_zip=True):
    """
    Extrae un archivo ZIP y opcionalmente lo elimina después.

    Args:
        zip_path: Ruta al archivo ZIP
        extract_path: Ruta donde extraer
        remove_zip: Si se debe eliminar el ZIP después de extraer

    Returns:
        list: Lista de archivos extraídos
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Filtrar solo archivos 3D
            model_files = [f for f in zip_ref.namelist()
                           if f.lower().endswith(('.stl', '.obj', '.ply'))]

            # Extraer solo los archivos 3D
            for file in model_files:
                zip_ref.extract(file, extract_path)

            print(f"Extraídos {len(model_files)} archivos 3D de {zip_path}")

        # Eliminar ZIP si se solicita
        if remove_zip and os.path.exists(zip_path):
            os.remove(zip_path)

        return model_files

    except zipfile.BadZipFile:
        print(f"Error: Archivo ZIP dañado o inválido: {zip_path}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return []
    except Exception as e:
        print(f"Error al extraer {zip_path}: {str(e)}")
        return []


def collect_thingiverse_models(api_token, search_term="vase", category="decor", min_likes=5,
                               max_models=100, output_dir="data/thingiverse"):
    """
    Recolecta modelos 3D de Thingiverse usando su API.

    Args:
        api_token: Token de API de Thingiverse
        search_term: Término de búsqueda
        category: Categoría de modelos
        min_likes: Número mínimo de likes para filtrar
        max_models: Número máximo de modelos a descargar
        output_dir: Directorio de salida

    Returns:
        int: Número de modelos descargados correctamente
    """
    # Crear sesión con encabezados de autenticación
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_token}",
        "User-Agent": DEFAULT_USER_AGENT
    })

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Archivo para guardar metadatos
    metadata_file = os.path.join(output_dir, "metadata.json")
    metadata = []

    # URL de búsqueda
    search_url = f"{THINGIVERSE_API_BASE}/search/{search_term}"
    params = {
        "per_page": 20,
        "sort": "popularity",
        "category": category
    }

    models_downloaded = 0
    page = 1

    # Continuar hasta alcanzar max_models o no haber más resultados
    while models_downloaded < max_models:
        params["page"] = page

        try:
            # Realizar solicitud
            response = session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            search_results = response.json()

            # Verificar si hay resultados
            if not search_results or "hits" not in search_results or not search_results["hits"]:
                print(f"No hay más resultados en la página {page}")
                break

            # Procesar cada resultado
            for item in search_results["hits"]:
                # Verificar likes
                like_count = item.get("like_count", 0)
                if like_count < min_likes:
                    continue

                # Obtener ID y nombre
                thing_id = item.get("id")
                name = item.get("name", f"thing_{thing_id}")

                print(f"Procesando: {name} (ID: {thing_id}, Likes: {like_count})")

                try:
                    # Obtener detalles del modelo
                    thing_url = f"{THINGIVERSE_API_BASE}/things/{thing_id}"
                    thing_response = session.get(thing_url, timeout=30)
                    thing_response.raise_for_status()
                    thing_data = thing_response.json()

                    # Guardar metadatos
                    metadata_item = {
                        "id": thing_id,
                        "name": name,
                        "description": thing_data.get("description", ""),
                        "creator": thing_data.get("creator", {}).get("name", "unknown"),
                        "like_count": like_count,
                        "download_count": thing_data.get("download_count", 0),
                        "category": category,
                        "date_added": thing_data.get("added", ""),
                        "license": thing_data.get("license", ""),
                        "tags": thing_data.get("tags", [])
                    }

                    # Obtener archivos para descargar
                    files_url = f"{THINGIVERSE_API_BASE}/things/{thing_id}/files"
                    files_response = session.get(files_url, timeout=30)
                    files_response.raise_for_status()
                    files_data = files_response.json()

                    # Crear directorio para este modelo
                    model_dir = os.path.join(output_dir, f"{thing_id}_{name.replace(' ', '_')[:30]}")
                    os.makedirs(model_dir, exist_ok=True)

                    # Descargar archivos
                    downloaded_files = []
                    has_3d_files = False

                    for file_data in files_data:
                        # Solo descargar archivos 3D
                        file_name = file_data.get("name", "")
                        file_url = file_data.get("download_url", "")

                        if not file_url or not file_name:
                            continue

                        # Verificar si es un archivo 3D o ZIP
                        is_3d = any(file_name.lower().endswith(ext) for ext in ['.stl', '.obj', '.ply'])
                        is_zip = file_name.lower().endswith('.zip')

                        if is_3d or is_zip:
                            file_path = os.path.join(model_dir, file_name)

                            # Descargar archivo
                            if download_file(file_url, file_path, session):
                                downloaded_files.append(file_path)

                                # Si es ZIP, extraerlo
                                if is_zip:
                                    extracted_files = extract_zip(file_path, model_dir)
                                    if any(f.lower().endswith(('.stl', '.obj', '.ply')) for f in extracted_files):
                                        has_3d_files = True
                                else:
                                    has_3d_files = True

                    # Si no se descargaron archivos 3D, eliminar directorio
                    if not has_3d_files:
                        print(f"No se encontraron archivos 3D para {name}, eliminando directorio")
                        shutil.rmtree(model_dir, ignore_errors=True)
                        continue

                    # Guardar metadatos de este modelo
                    metadata_item["files"] = downloaded_files
                    metadata.append(metadata_item)

                    models_downloaded += 1
                    print(f"Descargado modelo {models_downloaded}/{max_models}: {name}")

                    # Verificar si se alcanzó el límite
                    if models_downloaded >= max_models:
                        break

                    # Esperar un poco para no sobrecargar la API
                    time.sleep(1)

                except Exception as e:
                    print(f"Error procesando modelo {name}: {str(e)}")

            # Pasar a la siguiente página
            page += 1

            # Guardar metadatos hasta ahora
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Pequeña pausa entre páginas
            time.sleep(2)

        except Exception as e:
            print(f"Error obteniendo resultados de búsqueda (página {page}): {str(e)}")
            break

    print(f"Proceso completado. Se descargaron {models_downloaded} modelos en {output_dir}")
    return models_downloaded


def collect_free3d_models(search_term="vase", category="decor", max_models=100, output_dir="data/free3d"):
    """
    Recolecta modelos 3D de Free3D mediante web scraping simple.

    Args:
        search_term: Término de búsqueda
        category: Categoría de modelos
        max_models: Número máximo de modelos a descargar
        output_dir: Directorio de salida

    Returns:
        int: Número de modelos descargados correctamente
    """
    print("NOTA: Web scraping de Free3D no implementado debido a posibles restricciones legales.")
    print("Se recomienda utilizar su API oficial o descargar manualmente los modelos.")
    return 0


def organize_models_by_category(input_dir, output_dir):
    """
    Organiza los modelos descargados en categorías.

    Args:
        input_dir: Directorio de entrada con modelos sin organizar
        output_dir: Directorio de salida para modelos organizados

    Returns:
        dict: Estadísticas de organización (modelos por categoría)
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Leer archivo de metadatos si existe
    metadata_file = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"Error: No se encontró el archivo de metadatos en {input_dir}")
        return {}

    # Cargar metadatos
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Estadísticas
    stats = {}

    # Procesar cada modelo
    for item in tqdm(metadata, desc="Organizando modelos"):
        # Obtener categoría y etiquetas
        category = item.get("category", "uncategorized")
        tags = item.get("tags", [])

        # Refinar categoría basándose en etiquetas si está en "uncategorized"
        if category == "uncategorized" and tags:
            # Mapeo de etiquetas a categorías
            category_keywords = {
                "vase": ["vase", "flower", "plant", "container", "pot"],
                "furniture": ["furniture", "chair", "table", "desk", "shelf", "cabinet"],
                "decoration": ["decoration", "decor", "ornament", "figurine", "statue"],
                "household": ["household", "kitchen", "bathroom", "utility", "gadget"],
                "art": ["art", "sculpture", "artistic", "decorative", "abstract"]
            }

            # Buscar coincidencias
            for cat, keywords in category_keywords.items():
                if any(kw in tag.lower() for tag in tags for kw in keywords):
                    category = cat
                    break

        # Crear directorio de categoría
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        # Actualizar estadísticas
        stats[category] = stats.get(category, 0) + 1

        # Origen del modelo
        model_id = item.get("id")
        model_name = item.get("name", f"model_{model_id}").replace(" ", "_")[:30]
        source_dir = os.path.join(input_dir, f"{model_id}_{model_name}")

        if not os.path.exists(source_dir):
            print(f"Advertencia: No se encontró el directorio origen {source_dir}")
            continue

        # Destino del modelo
        dest_dir = os.path.join(category_dir, f"{model_id}_{model_name}")

        # Copiar archivos
        if not os.path.exists(dest_dir):
            try:
                shutil.copytree(source_dir, dest_dir)
            except Exception as e:
                print(f"Error copiando {source_dir} a {dest_dir}: {str(e)}")

    # Guardar estadísticas
    stats_file = os.path.join(output_dir, "category_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Organización completada. Estadísticas guardadas en {stats_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Recolectar modelos 3D de varias fuentes")

    # Fuente de datos
    parser.add_argument('--source', type=str, default='thingiverse',
                        choices=['thingiverse', 'free3d', 'organize'],
                        help='Fuente de los modelos')

    # Parámetros generales
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directorio de salida')
    parser.add_argument('--max_models', type=int, default=100,
                        help='Número máximo de modelos a descargar')

    # Parámetros específicos de Thingiverse
    parser.add_argument('--api_token', type=str,
                        help='Token de API de Thingiverse')
    parser.add_argument('--search_term', type=str, default='vase',
                        help='Término de búsqueda')
    parser.add_argument('--category', type=str, default='decor',
                        help='Categoría de modelos')
    parser.add_argument('--min_likes', type=int, default=5,
                        help='Número mínimo de likes')

    # Parámetros para organización
    parser.add_argument('--input_dir', type=str,
                        help='Directorio de entrada para organización')

    args = parser.parse_args()

    # Ejecutar acción según la fuente
    if args.source == 'thingiverse':
        if not args.api_token:
            print("Error: Se requiere token de API para Thingiverse")
            parser.print_help()
            return

        collect_thingiverse_models(
            api_token=args.api_token,
            search_term=args.search_term,
            category=args.category,
            min_likes=args.min_likes,
            max_models=args.max_models,
            output_dir=os.path.join(args.output_dir, 'thingiverse')
        )

    elif args.source == 'free3d':
        collect_free3d_models(
            search_term=args.search_term,
            category=args.category,
            max_models=args.max_models,
            output_dir=os.path.join(args.output_dir, 'free3d')
        )

    elif args.source == 'organize':
        if not args.input_dir:
            print("Error: Se requiere directorio de entrada para organización")
            parser.print_help()
            return

        organize_models_by_category(
            input_dir=args.input_dir,
            output_dir=os.path.join(args.output_dir, 'organized')
        )


if __name__ == "__main__":
    main()