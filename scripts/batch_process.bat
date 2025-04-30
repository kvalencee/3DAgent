@echo off
set BLENDER_PATH="C:\Program Files\Blender Foundation\Blender\blender.exe"
set PYTHON_SCRIPT="C:\ruta_al_proyecto\download_sketchfab.py"
set NORMALIZE_SCRIPT="C:\ruta_al_proyecto\normalize_glb.py"

:: Paso 1: Descargar modelos
python %PYTHON_SCRIPT%

:: Paso 2: Normalizar todos los GLB descargados
for %%f in ("C:\ruta_al_proyecto\vases_ai\*.glb") do (
    %BLENDER_PATH% --background --python %NORMALIZE_SCRIPT% -- "%%f" "%%~nf_normalized.glb"
    del "%%f"  # Opcional: eliminar original
)