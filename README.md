### Jazz-Inador

 Proyecto final de data science. Proyecto para filtrar musica en tiempo real dentro de una cola de reproduccion compartida, filtrando el genero de la musica recomendada por un sistema de Neural Network o Random Forest. Si la musica recomendada no es de el genero seleccionado (JAZZ) esta no se reproduce, siendo ignorada en la cola de reproduccion.

### Data: 

| Genres    | Samples | Espectrograms | Features |
| --------- | ------- | ------------- | -------- |
| blues     | 100     | 1000          | 1000     |
| classical | 100     | 1000          | 1000     |
| country   | 100     | 1000          | 1000     |
| disco     | 100     | 1000          | 1000     |
| hiphop    | 100     | 1000          | 1000     |
| jazz      | 99      | 999           | 999      |
| metal     | 100     | 1000          | 1000     |
| pop       | 100     | 1000          | 1000     |
| reggae    | 100     | 1000          | 1000     |
| rock      | 100     | 1000          | 1000     |
| **Total** | **999** | **9999**      | **9999** |
### Commands:

`Instalar Dependencias:`

```python
uv sync
```

`Generar Espectrogramas:`

```bash
uv run Data-generator/wav-to-mel-spectrograms.py
```

`Generar CSV de Features:`

```bash
uv run Data-generator/features-to-csv.py
```

`Validar Data:`

```bash
uv run Data-generator/validate_dataset.py
```

### Bibliography

- Link to zip file: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data?select=Data
- https://medium.com/@niknair31898/music-genre-classification-using-mel-spectrograms-88ce05400d3c