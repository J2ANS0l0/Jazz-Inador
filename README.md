# Jazz-Inador

Final data science project. It filters music in real time within a shared playback queue, classifying the genre of recommended songs with a CNN (convolutional neural network) or Random Forest model. If a recommended song is not of the selected genre (JAZZ), it is not played and is skipped in the playback queue.

### Data: 

| Genres    | Samples | Spectrograms | Features |
| --------- | ------- | ------------ | -------- |
| blues     | 100     | 1000         | 1000     |
| classical | 100     | 1000         | 1000     |
| country   | 100     | 1000         | 1000     |
| disco     | 100     | 1000         | 1000     |
| hiphop    | 100     | 1000         | 1000     |
| jazz      | 99      | 990          | 990      |
| metal     | 100     | 1000         | 1000     |
| pop       | 100     | 1000         | 1000     |
| reggae    | 100     | 1000         | 1000     |
| rock      | 100     | 1000         | 1000     |
| **Total** | **999** | **9990**     | **9990** |
### Commands:

####  Install dependencies:

```bash
uv sync
# Optional: also install PyTorch (CUDA) to train the CNN on an NVIDIA GPU.
# Not needed to use the trained models.
uv sync --group gpu
```
#### Generate data

```bash
# Generate spectrograms:
uv run data_generator/wav_to_mel_spectrograms.py
# Generate features CSV:
uv run data_generator/features_to_csv.py
# Validate data:
uv run data_generator/validate_dataset.py
```
#### Train models
```bash
# Train random forest
uv run random_forest/random_forest_genres.py
# Train CNN (NumPy, CPU)
uv run cnn/cnn_genres.py
# Train CNN (PyTorch, NVIDIA GPU with CUDA 13; needs uv sync --group gpu)
uv run --group gpu cnn/cnn_genres_torch.py
```

#### Classify audio files using each model

```bash
# Classify an audio file with the random forest:
uv run random_forest/random_forest_genres_classifier.py <audio_file>
# Classify an audio file with the CNN (models/cnn.json):
uv run cnn/cnn_genres_classifier.py <audio_file>
```

### Bibliography

- Link to zip file: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data?select=Data
- https://medium.com/@niknair31898/music-genre-classification-using-mel-spectrograms-88ce05400d3c