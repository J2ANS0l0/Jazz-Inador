import argparse
import json
from functools import lru_cache
from pathlib import Path

import librosa
import librosa.display
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

MODEL_RELATIVE_PATH = Path("models") / "cnn.json"
BATCH_SIZE = 16

# Same configuration as data_generator/wav_to_mel_spectrograms.py.
# That script loads GTZAN with sr=None, which is 22050 Hz for every file,
# so any input audio is resampled to 22050 Hz to match the training data.
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
N_FFT = 8192
HOP_LENGTH = 128
WINDOW = "hamming"
N_MELS = 192
FMIN = 0.0
FMAX = None
HTK = False
MEL_NORM = "slaney"
POWER = 2.0
TOP_DB = 75
CMAP = "magma"
FIGSIZE = (4.32, 2.88)
DPI = 100
IMAGE_WIDTH = 432
IMAGE_HEIGHT = 288


def main():
    parser = argparse.ArgumentParser(description="Classify the genre of an audio file with the CNN.")
    parser.add_argument("audio", type=Path, help="Path to the audio file")
    parser.add_argument("--model", type=Path, default=None, help=f"Model JSON (default: {MODEL_RELATIVE_PATH})")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")
    audio = args.audio
    spectrograms = audio_to_spectrograms(audio)
    result = cnn_genre_classifier(spectrograms, args.model)
    print(f"Segments: {len(spectrograms)}")
    for genre, probability in result.items():
        print(f"{genre}: {probability:.4f}")


def _render_spectrogram(segment, sr):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=HTK,
        norm=MEL_NORM,
        power=POWER,
    )
    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max, top_db=TOP_DB)

    fig = Figure(figsize=FIGSIZE, dpi=DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    librosa.display.specshow(mel_spectrogram_db, sr=sr, cmap=CMAP, ax=ax)
    ax.axis("off")
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def audio_to_spectrograms(audio_path):
    y, sr = librosa.load(Path(audio_path), sr=SAMPLE_RATE, mono=True)
    segment_length = SEGMENT_SECONDS * sr
    segment_count = max(1, len(y) // segment_length)
    y = librosa.util.fix_length(y, size=segment_count * segment_length)
    return np.stack(
        [
            _render_spectrogram(y[fragment * segment_length:(fragment + 1) * segment_length], sr)
            for fragment in range(segment_count)
        ]
    )


def load_spectrogram_png(png_path):
    with Image.open(png_path) as image:
        return np.asarray(image.convert("RGB"))


def _default_model_path():
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for folder in [start, *start.parents]:
        candidate = folder / MODEL_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{MODEL_RELATIVE_PATH} not found from {start}")


def _resolve_model_path(model_path):
    path = Path(model_path) if model_path else _default_model_path()
    return str(path.resolve())


def _load_layer(layer):
    layer = dict(layer)
    for key in ("weight", "bias", "running_mean", "running_var"):
        if key in layer:
            layer[key] = np.asarray(layer[key], dtype=np.float32)
    return layer


@lru_cache(maxsize=None)
def _load_model(model_path):
    with open(model_path, encoding="utf-8") as f:
        model = json.load(f)["model"]
    model_input = dict(model["input"])
    model_input["mean"] = np.asarray(model_input["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    model_input["std"] = np.asarray(model_input["std"], dtype=np.float32).reshape(1, -1, 1, 1)
    return tuple(model["classes"]), model_input, tuple(_load_layer(layer) for layer in model["layers"])


def _to_batch(spectrograms):
    images = np.asarray(spectrograms)
    if images.ndim == 3:
        images = images[np.newaxis]
    if images.ndim != 4 or images.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(f"Expected spectrograms with shape (N, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3), got {images.shape}")
    if images.shape[0] == 0:
        raise ValueError("No spectrograms provided")
    if images.dtype != np.uint8:
        raise ValueError("Spectrograms must be uint8 RGB images")
    return images


def preprocess(spectrograms, model_input):
    size = (model_input["width"], model_input["height"])
    resized = np.stack(
        [np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BOX)) for image in _to_batch(spectrograms)]
    )
    X = resized.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return (X - model_input["mean"]) / model_input["std"]


def _conv2d(X, layer):
    pad_h, pad_w = layer["padding"]
    stride_h, stride_w = layer["stride"]
    kernel_h, kernel_w = layer["kernel_size"]
    X = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    windows = np.lib.stride_tricks.sliding_window_view(X, (kernel_h, kernel_w), axis=(2, 3))
    windows = windows[:, :, ::stride_h, ::stride_w]
    out = np.tensordot(windows, layer["weight"], axes=([1, 4, 5], [1, 2, 3]))
    return out.transpose(0, 3, 1, 2) + layer["bias"].reshape(1, -1, 1, 1)


def _batch_norm2d(X, layer):
    scale = layer["weight"] / np.sqrt(layer["running_var"] + layer["eps"])
    shift = layer["bias"] - layer["running_mean"] * scale
    return X * scale.reshape(1, -1, 1, 1) + shift.reshape(1, -1, 1, 1)


def _max_pool2d(X, layer):
    kernel_h, kernel_w = layer["kernel_size"]
    n, c, h, w = X.shape
    X = X[:, :, : h - h % kernel_h, : w - w % kernel_w]
    return X.reshape(n, c, h // kernel_h, kernel_h, w // kernel_w, kernel_w).max(axis=(3, 5))


def _softmax(X):
    exp = np.exp(X - X.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


LAYER_FUNCTIONS = {
    "conv2d": _conv2d,
    "batch_norm2d": _batch_norm2d,
    "relu": lambda X, layer: np.maximum(X, 0.0),
    "max_pool2d": _max_pool2d,
    "mean": lambda X, layer: X.mean(axis=layer["axis"]),
    "flatten": lambda X, layer: X.reshape(X.shape[0], -1),
    "dropout": lambda X, layer: X,
    "linear": lambda X, layer: X @ layer["weight"].T + layer["bias"],
    "softmax": lambda X, layer: _softmax(X),
}


def forward(X, layers):
    for layer in layers:
        X = LAYER_FUNCTIONS[layer["type"]](X, layer)
    return X


def predict_proba(spectrograms, model_path=None):
    classes, model_input, layers = _load_model(_resolve_model_path(model_path))
    images = _to_batch(spectrograms)
    proba = [
        forward(preprocess(images[start:start + BATCH_SIZE], model_input), layers)
        for start in range(0, len(images), BATCH_SIZE)
    ]
    proba = np.concatenate(proba).astype(np.float64)
    # Renormalize in float64 so each row sums to 1 at float64 precision (the network computes in float32).
    return classes, proba / proba.sum(axis=1, keepdims=True)


def cnn_genre_classifier(spectrograms, model_path=None):
    classes, proba = predict_proba(spectrograms, model_path)
    genre_proba = proba.mean(axis=0)
    return dict(sorted(zip(classes, genre_proba.tolist()), key=lambda item: item[1], reverse=True))


def audio_to_genre(audio_path, model_path=None):
    return cnn_genre_classifier(audio_to_spectrograms(audio_path), model_path)


if __name__ == "__main__":
    main()
