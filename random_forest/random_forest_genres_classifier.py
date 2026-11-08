import json
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

MODEL_RELATIVE_PATH = Path("models") / "random_forest.json"
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
MFCC_COUNT = 20

COLUMNS = [
    "filename",
    "length",
    "chroma_stft_mean",
    "chroma_stft_var",
    "rms_mean",
    "rms_var",
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_var",
    "rolloff_mean",
    "rolloff_var",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_var",
    "harmony_mean",
    "harmony_var",
    "perceptr_mean",
    "perceptr_var",
    "tempo",
]
for _index in range(1, MFCC_COUNT + 1):
    COLUMNS.extend([f"mfcc{_index}_mean", f"mfcc{_index}_var"])
COLUMNS.append("label")


def main():
    audio = r"PATH"
    features = extract_features(audio)
    result = random_forest_genre_classifier(features)
    print(f"Segments: {len(features['filename'])}")
    for genre, probability in result.items():
        print(f"{genre}: {probability:.4f}")


def _segment_features(segment, sr):
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y=segment)
    harmonic, percussive = librosa.effects.hpss(segment)
    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=MFCC_COUNT)

    values = {
        "length": len(segment),
        "chroma_stft_mean": chroma.mean(),
        "chroma_stft_var": chroma.var(),
        "rms_mean": rms.mean(),
        "rms_var": rms.var(),
        "spectral_centroid_mean": centroid.mean(),
        "spectral_centroid_var": centroid.var(),
        "spectral_bandwidth_mean": bandwidth.mean(),
        "spectral_bandwidth_var": bandwidth.var(),
        "rolloff_mean": rolloff.mean(),
        "rolloff_var": rolloff.var(),
        "zero_crossing_rate_mean": zero_crossing.mean(),
        "zero_crossing_rate_var": zero_crossing.var(),
        "harmony_mean": harmonic.mean(),
        "harmony_var": harmonic.var(),
        "perceptr_mean": percussive.mean(),
        "perceptr_var": percussive.var(),
        "tempo": float(np.atleast_1d(tempo)[0]),
    }
    for index in range(MFCC_COUNT):
        values[f"mfcc{index + 1}_mean"] = mfcc[index].mean()
        values[f"mfcc{index + 1}_var"] = mfcc[index].var()
    return values


def extract_features(audio_path):
    audio_path = Path(audio_path)
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    segment_length = SEGMENT_SECONDS * sr
    segment_count = max(1, len(y) // segment_length)
    y = librosa.util.fix_length(y, size=segment_count * segment_length)

    features = {column: [] for column in COLUMNS}
    for fragment in range(segment_count):
        segment = y[fragment * segment_length:(fragment + 1) * segment_length]
        values = _segment_features(segment, sr)
        features["filename"].append(f"{audio_path.stem}{fragment:03d}")
        for key, value in values.items():
            features[key].append(value.item() if hasattr(value, "item") else value)
        features["label"].append(None)
    return features


def _default_model_path():
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for folder in [start, *start.parents]:
        candidate = folder / MODEL_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{MODEL_RELATIVE_PATH} not found from {start}")


@lru_cache(maxsize=None)
def _load_model(model_path):
    with open(model_path, encoding="utf-8") as f:
        model = json.load(f)["model"]
    trees = []
    for tree in model["trees"]:
        left = np.asarray(tree["children_left"], dtype=np.int64)
        is_leaf = left == -1
        leaf_index = np.full(left.shape[0], -1, dtype=np.int64)
        leaf_index[is_leaf] = np.arange(is_leaf.sum())
        trees.append(
            (
                left,
                np.asarray(tree["children_right"], dtype=np.int64),
                np.asarray(tree["feature"], dtype=np.int64),
                np.asarray(tree["threshold"], dtype=np.float64),
                leaf_index,
                np.asarray(tree["leaf_values"], dtype=np.float64),
            )
        )
    return tuple(model["classes"]), tuple(model["feature_names"]), tuple(trees)


def _to_matrix(features, feature_names):
    if isinstance(features, dict):
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        columns = [np.atleast_1d(np.asarray(features[name], dtype=np.float64)) for name in feature_names]
        if len({column.shape for column in columns}) != 1 or columns[0].ndim != 1:
            raise ValueError("All features must have the same number of fragments")
        X = np.column_stack(columns)
    else:
        fragments = list(features)
        missing = sorted({name for fragment in fragments for name in feature_names if name not in fragment})
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X = np.asarray([[fragment[name] for name in feature_names] for fragment in fragments], dtype=np.float64)
    if X.shape[0] == 0:
        raise ValueError("No fragments provided")
    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values")
    return X.astype(np.float32)


def _predict_proba(X, trees, n_classes):
    rows = np.arange(X.shape[0])
    proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    for left, right, feature, threshold, leaf_index, leaf_values in trees:
        node = np.zeros(X.shape[0], dtype=np.int64)
        active = left[node] != -1
        while active.any():
            current = node[active]
            go_left = X[rows[active], feature[current]] <= threshold[current]
            node[active] = np.where(go_left, left[current], right[current])
            active = left[node] != -1
        proba += leaf_values[leaf_index[node]]
    return proba / len(trees)


def random_forest_genre_classifier(features, model_path=None):
    path = Path(model_path) if model_path else _default_model_path()
    classes, feature_names, trees = _load_model(str(path.resolve()))
    X = _to_matrix(features, feature_names)
    genre_proba = _predict_proba(X, trees, len(classes)).mean(axis=0)
    return dict(sorted(zip(classes, genre_proba.tolist()), key=lambda item: item[1], reverse=True))


def audio_to_genre(audio_path, model_path=None):
    return random_forest_genre_classifier(extract_features(audio_path), model_path)


if __name__ == "__main__":
    main()