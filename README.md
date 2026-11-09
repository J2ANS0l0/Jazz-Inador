# Jazz-Inador

Final data science project. It filters music in real time within a shared playback queue, classifying the genre of recommended songs with a Neural Network or Random Forest model. If a recommended song is not of the selected genre (JAZZ), it is not played and is skipped in the playback queue.

### Data: 

| Genres    | Samples | Spectrograms | Features |
| --------- | ------- | ------------ | -------- |
| blues     | 100     | 1000         | 1000     |
| classical | 100     | 1000         | 1000     |
| country   | 100     | 1000         | 1000     |
| disco     | 100     | 1000         | 1000     |
| hiphop    | 100     | 1000         | 1000     |
| jazz      | 99      | 999          | 999      |
| metal     | 100     | 1000         | 1000     |
| pop       | 100     | 1000         | 1000     |
| reggae    | 100     | 1000         | 1000     |
| rock      | 100     | 1000         | 1000     |
| **Total** | **999** | **9999**     | **9999** |
### Commands:

####  Install dependencies:

```bash
uv sync
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
# Train convolutional neural network (NumPy, CPU)
uv run convolutional_neural_network/convolutional_neural_network_genres.py
# Train convolutional neural network (PyTorch, NVIDIA GPU with CUDA 13)
uv run convolutional_neural_network/convolutional_neural_network_genres_torch.py
```

#### Classify audio files using each model

```bash
# Classify an audio file with the random forest:
uv run random_forest/random_forest_genres_classifier.py <audio_file>
# Classify an audio file with the convolutional neural network:
uv run convolutional_neural_network/convolutional_neural_network_genres_classifier.py <audio_file>
# Classify an audio file with the PyTorch-trained network:
uv run convolutional_neural_network/convolutional_neural_network_genres_classifier.py <audio_file> --model models/convolutional_neural_network_torch.json
```

### Bibliography

- Link to zip file: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data?select=Data
- https://medium.com/@niknair31898/music-genre-classification-using-mel-spectrograms-88ce05400d3c