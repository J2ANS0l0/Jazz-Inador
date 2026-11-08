# Jazz-Inador

 Proyecto final de data science. Proyecto para filtrar musica en tiempo real dentro de una cola de reproduccion compartida, filtrando el genero de la musica recomendada por un sistema de Neural Network o Random Forest. Si la musica recomendada no es de el genero seleccionado (JAZZ) esta no se reproduce, siendo ignorada en la cola de reproduccion.

## Data: 

| filename | samples | espectrograms |
|---|---:|---:|
| blues | 100 | 1000 |
| classical | 100 | 1000 |
| country | 100 | 1000 |
| disco | 100 | 1000 |
| hiphop | 100 | 1000 |
| jazz | 99 | 999 |
| metal | 100 | 1000 |
| pop | 100 | 1000 |
| reggae | 100 | 1000 |
| rock | 100 | 1000 |
| **Total** | **999** | **9999** |

## Commands:

Download ZIP file: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data?select=Data

uv init --bare
uv python install 3.13
uv python pin 3.13

GENERAR DATA DE MUESTRAS MUSICALES:
uv run --with librosa --with matplotlib --with numpy wav-to-mel-spectrograms.py
uv run --with soundfile --with pillow validate_all.py

VALIDAR DATA:
uv run --with soundfile --with pillow validate_all.py