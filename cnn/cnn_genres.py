import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/mel_spectrograms"
MODEL_PATH = "models/cnn_numpy.json"
CLASSIFIER_PATH = Path(__file__).resolve().parent / "cnn_genres_classifier.py"
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.2
VALIDATION_FRACTION = 0.15
SONG_ID_PATTERN = r"([a-z]+)(\d{5})(\d{3})"

IMAGE_WIDTH = 432
IMAGE_HEIGHT = 288
INPUT_WIDTH = 216
INPUT_HEIGHT = 144
INPUT_CHANNELS = 3

CONV_FILTERS = [16, 32, 64, 128]
KERNEL_SIZE = 3
POOL_SIZE = 2
DENSE_UNITS = 64
DROPOUT = 0.5
BN_EPS = 1e-5
BN_MOMENTUM = 0.1

EPOCHS = 40
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
LR_FACTOR = 0.5
LR_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 8
MAX_TIME_MASK = 24
MAX_FREQUENCY_MASK = 16


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Layers work in NHWC layout (batch, frequency, time, channels) during training.
# They are exported in the NCHW layout used by cnn_genres_classifier.py.


class Layer:
    role = "hidden"

    def __init__(self, name):
        self.name = name
        self.params = {}
        self.grads = {}
        self.buffers = {}
        self.decay = set()

    def parameter_count(self):
        return int(sum(value.size for value in self.params.values()))


class Conv2D(Layer):
    def __init__(self, name, in_channels, out_channels, rng, input_gradient=True):
        super().__init__(name)
        fan_in = KERNEL_SIZE * KERNEL_SIZE * in_channels
        self.params["weight"] = (rng.standard_normal((KERNEL_SIZE, KERNEL_SIZE, in_channels, out_channels)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.params["bias"] = np.zeros(out_channels, dtype=np.float32)
        self.decay = {"weight"}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_gradient = input_gradient

    def forward(self, X, training=False, rng=None):
        n, h, w, c = X.shape
        pad = KERNEL_SIZE // 2
        Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        cols = np.concatenate(
            [Xp[:, i:i + h, j:j + w, :] for i in range(KERNEL_SIZE) for j in range(KERNEL_SIZE)], axis=3
        ).reshape(n * h * w, -1)
        out = cols @ self.params["weight"].reshape(-1, self.out_channels) + self.params["bias"]
        if training:
            self.cache = (cols, X.shape)
        return out.reshape(n, h, w, self.out_channels)

    def backward(self, dout):
        cols, (n, h, w, c) = self.cache
        self.cache = None
        dflat = dout.reshape(-1, self.out_channels)
        self.grads["weight"] = (cols.T @ dflat).reshape(self.params["weight"].shape)
        self.grads["bias"] = dflat.sum(axis=0)
        if not self.input_gradient:
            return None
        dcols = (dflat @ self.params["weight"].reshape(-1, self.out_channels).T).reshape(n, h, w, KERNEL_SIZE * KERNEL_SIZE, c)
        pad = KERNEL_SIZE // 2
        dXp = np.zeros((n, h + 2 * pad, w + 2 * pad, c), dtype=dout.dtype)
        for k in range(KERNEL_SIZE * KERNEL_SIZE):
            i, j = divmod(k, KERNEL_SIZE)
            dXp[:, i:i + h, j:j + w, :] += dcols[:, :, :, k, :]
        return dXp[:, pad:pad + h, pad:pad + w, :]

    def export(self):
        return {
            "type": "conv2d",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": [KERNEL_SIZE, KERNEL_SIZE],
            "stride": [1, 1],
            "padding": [KERNEL_SIZE // 2, KERNEL_SIZE // 2],
            "weight": self.params["weight"].transpose(3, 2, 0, 1),
            "bias": self.params["bias"],
        }


class BatchNorm2D(Layer):
    def __init__(self, name, channels):
        super().__init__(name)
        self.params["weight"] = np.ones(channels, dtype=np.float32)
        self.params["bias"] = np.zeros(channels, dtype=np.float32)
        self.buffers["running_mean"] = np.zeros(channels, dtype=np.float32)
        self.buffers["running_var"] = np.ones(channels, dtype=np.float32)
        self.channels = channels

    def forward(self, X, training=False, rng=None):
        X2 = X.reshape(-1, self.channels)
        if training:
            # Channel reductions as matrix-vector products (multithreaded BLAS) instead of axis sums.
            ones = np.ones(X2.shape[0], dtype=X.dtype)
            mean = (ones @ X2) / X2.shape[0]
            centered = X2 - mean
            var = np.einsum("ij,ij->j", centered, centered) / X2.shape[0]
            count = X2.shape[0]
            self.buffers["running_mean"] = ((1 - BN_MOMENTUM) * self.buffers["running_mean"] + BN_MOMENTUM * mean).astype(np.float32)
            self.buffers["running_var"] = ((1 - BN_MOMENTUM) * self.buffers["running_var"] + BN_MOMENTUM * var * count / (count - 1)).astype(np.float32)
        else:
            centered = X2 - self.buffers["running_mean"]
            var = self.buffers["running_var"]
        inv_std = (1.0 / np.sqrt(var + BN_EPS)).astype(X.dtype)
        if training:
            self.cache = (centered, inv_std, X.shape)
        out = centered * (self.params["weight"] * inv_std)
        out += self.params["bias"]
        return out.reshape(X.shape)

    def backward(self, dout):
        centered, inv_std, shape = self.cache
        self.cache = None
        dout2 = dout.reshape(-1, self.channels)
        count = dout2.shape[0]
        ones = np.ones(count, dtype=dout.dtype)
        dout_sum = ones @ dout2
        dout_centered = np.einsum("ij,ij->j", dout2, centered)
        self.grads["weight"] = dout_centered * inv_std
        self.grads["bias"] = dout_sum
        # dX = gamma * inv_std * (dout - mean(dout) - X_hat * mean(dout * X_hat)), expanded per channel.
        scale = self.params["weight"] * inv_std
        dX = dout2 * scale
        dX -= centered * (scale * inv_std * inv_std * dout_centered / count)
        dX -= scale * dout_sum / count
        return dX.reshape(shape)

    def export(self):
        return {
            "type": "batch_norm2d",
            "num_features": self.channels,
            "eps": BN_EPS,
            "momentum": BN_MOMENTUM,
            "weight": self.params["weight"],
            "bias": self.params["bias"],
            "running_mean": self.buffers["running_mean"],
            "running_var": self.buffers["running_var"],
        }


class ReLU(Layer):
    def forward(self, X, training=False, rng=None):
        out = np.maximum(X, 0)
        if training:
            self.cache = out
        return out

    def backward(self, dout):
        out, self.cache = self.cache, None
        return np.multiply(dout, out > 0)

    def export(self):
        return {"type": "relu"}


class MaxPool2D(Layer):
    def forward(self, X, training=False, rng=None):
        n, h, w, c = X.shape
        h_out, w_out = h // POOL_SIZE, w // POOL_SIZE
        blocks = X[:, : h_out * POOL_SIZE, : w_out * POOL_SIZE, :].reshape(n, h_out, POOL_SIZE, w_out, POOL_SIZE, c)
        out = blocks.max(axis=(2, 4))
        if training:
            self.cache = (blocks == out[:, :, np.newaxis, :, np.newaxis, :], X.shape)
        return out

    def backward(self, dout):
        mask, shape = self.cache
        self.cache = None
        n, h_out, _, w_out, _, c = mask.shape
        dX = np.zeros(shape, dtype=dout.dtype)
        dX[:, : h_out * POOL_SIZE, : w_out * POOL_SIZE, :] = (mask * dout[:, :, np.newaxis, :, np.newaxis, :]).reshape(
            n, h_out * POOL_SIZE, w_out * POOL_SIZE, c
        )
        return dX

    def export(self):
        return {"type": "max_pool2d", "kernel_size": [POOL_SIZE, POOL_SIZE], "stride": [POOL_SIZE, POOL_SIZE]}


class TimeMean(Layer):
    def forward(self, X, training=False, rng=None):
        if training:
            self.cache = X.shape
        return X.mean(axis=2)

    def backward(self, dout):
        shape, self.cache = self.cache, None
        return np.broadcast_to(dout[:, :, np.newaxis, :] / shape[2], shape).copy()

    def export(self):
        return {"type": "mean", "axis": 3, "dimension": "time"}


class Flatten(Layer):
    def forward(self, X, training=False, rng=None):
        if training:
            self.cache = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        shape, self.cache = self.cache, None
        return dout.reshape(shape)

    def export(self):
        return {"type": "flatten"}


class Dense(Layer):
    def __init__(self, name, in_features, out_features, rng, input_layout=None):
        super().__init__(name)
        self.params["weight"] = (rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)).astype(np.float32)
        self.params["bias"] = np.zeros(out_features, dtype=np.float32)
        self.decay = {"weight"}
        self.in_features = in_features
        self.out_features = out_features
        self.input_layout = input_layout

    def forward(self, X, training=False, rng=None):
        if training:
            self.cache = X
        return X @ self.params["weight"] + self.params["bias"]

    def backward(self, dout):
        X, self.cache = self.cache, None
        self.grads["weight"] = X.T @ dout
        self.grads["bias"] = dout.sum(axis=0)
        return dout @ self.params["weight"].T

    def export(self):
        weight = self.params["weight"].T
        if self.input_layout is not None:
            # Training flattens (frequency, channels); the classifier flattens (channels, frequency).
            frequency, channels = self.input_layout
            weight = weight.reshape(-1, frequency, channels).transpose(0, 2, 1).reshape(self.out_features, -1)
        return {
            "type": "linear",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight": weight,
            "bias": self.params["bias"],
        }


class Dropout(Layer):
    def __init__(self, name, p):
        super().__init__(name)
        self.p = p

    def forward(self, X, training=False, rng=None):
        if not training:
            return X
        self.cache = (rng.random(X.shape) >= self.p).astype(X.dtype) / (1.0 - self.p)
        return X * self.cache

    def backward(self, dout):
        mask, self.cache = self.cache, None
        return dout * mask

    def export(self):
        return {"type": "dropout", "p": self.p}


class Softmax(Layer):
    role = "output"

    def forward(self, X, training=False, rng=None):
        return softmax(X)

    def export(self):
        return {"type": "softmax"}


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class CNN:
    def __init__(self, n_classes, rng):
        self.layers = []
        channels, height, width = INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH
        for index, filters in enumerate(CONV_FILTERS, start=1):
            self.layers += [
                Conv2D(f"conv{index}", channels, filters, rng, input_gradient=index > 1),
                BatchNorm2D(f"batch_norm{index}", filters),
                ReLU(f"relu{index}"),
                MaxPool2D(f"max_pool{index}"),
            ]
            channels, height, width = filters, height // POOL_SIZE, width // POOL_SIZE
        output = Dense("output", DENSE_UNITS, n_classes, rng)
        output.role = "output"
        self.layers += [
            TimeMean("time_mean"),
            Flatten("flatten"),
            Dense("dense", height * channels, DENSE_UNITS, rng, input_layout=(height, channels)),
            ReLU("relu_dense"),
            Dropout("dropout", DROPOUT),
            output,
        ]
        self.softmax = Softmax("softmax")

    def forward(self, X, training=False, rng=None):
        for layer in self.layers:
            X = layer.forward(X, training, rng)
        return X

    def backward(self, dlogits):
        for layer in reversed(self.layers):
            dlogits = layer.backward(dlogits)

    def predict_proba(self, X):
        return softmax(self.forward(X))

    def state(self):
        return [({k: v.copy() for k, v in layer.params.items()}, {k: v.copy() for k, v in layer.buffers.items()}) for layer in self.layers]

    def load_state(self, state):
        for layer, (params, buffers) in zip(self.layers, state):
            layer.params = {k: v.copy() for k, v in params.items()}
            layer.buffers = {k: v.copy() for k, v in buffers.items()}

    def export(self):
        X = np.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), dtype=np.float32)
        layers = []
        for layer in [*self.layers, self.softmax]:
            X = layer.forward(X)
            shape = list(X.shape[1:])
            output_shape = [shape[-1], *shape[:-1]] if len(shape) > 1 else shape
            layers.append(
                {
                    "name": layer.name,
                    "role": layer.role,
                    **layer.export(),
                    "output_shape": output_shape,
                    "parameters": layer.parameter_count(),
                }
            )
        return layers


class AdamW:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        self.step_count = 0
        self.m = [{k: np.zeros_like(v) for k, v in layer.params.items()} for layer in layers]
        self.v = [{k: np.zeros_like(v) for k, v in layer.params.items()} for layer in layers]

    def step(self):
        self.step_count += 1
        beta1, beta2 = ADAM_BETAS
        correction1 = 1 - beta1 ** self.step_count
        correction2 = 1 - beta2 ** self.step_count
        for layer, m, v in zip(self.layers, self.m, self.v):
            for key, param in layer.params.items():
                grad = layer.grads[key]
                m[key] = beta1 * m[key] + (1 - beta1) * grad
                v[key] = beta2 * v[key] + (1 - beta2) * grad * grad
                if key in layer.decay:
                    param *= 1 - self.learning_rate * WEIGHT_DECAY
                param -= (self.learning_rate * (m[key] / correction1) / (np.sqrt(v[key] / correction2) + ADAM_EPS)).astype(np.float32)


def _load_image(path):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise ValueError(f"Unexpected size {image.size}: {path}")
        resized = image.convert("RGB").resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.uint8), hashlib.sha256(data).hexdigest()


def load_dataset(data_path=DATA_PATH):
    data_path = Path(data_path)
    paths = sorted(data_path.rglob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG spectrograms found in {data_path}")

    pattern = re.compile(SONG_ID_PATTERN)
    filenames, labels, song_ids = [], [], []
    for path in paths:
        match = pattern.fullmatch(path.stem)
        if not match or match.group(1) != path.parent.name:
            raise ValueError(f"{path} does not match {SONG_ID_PATTERN} inside its genre folder")
        filenames.append(path.stem)
        labels.append(path.parent.name)
        song_ids.append(match.group(1) + match.group(2))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        loaded = list(pool.map(_load_image, paths))
    images = np.stack([image for image, _ in loaded])
    digest = hashlib.sha256()
    for path, (_, file_hash) in zip(paths, loaded):
        digest.update(f"{path.relative_to(data_path).as_posix()}:{file_hash}\n".encode())

    return images, np.asarray(filenames), np.asarray(labels), np.asarray(song_ids), digest.hexdigest()


def _channel_statistics(images, chunk_size=512):
    total = np.zeros(INPUT_CHANNELS)
    total_sq = np.zeros(INPUT_CHANNELS)
    for start in range(0, len(images), chunk_size):
        chunk = images[start:start + chunk_size].astype(np.float64) / 255.0
        total += chunk.sum(axis=(0, 1, 2))
        total_sq += (chunk ** 2).sum(axis=(0, 1, 2))
    count = len(images) * INPUT_HEIGHT * INPUT_WIDTH
    mean = total / count
    return mean.astype(np.float32), np.sqrt(total_sq / count - mean ** 2).astype(np.float32)


def _normalize(images, mean, std):
    return (images.astype(np.float32) / 255.0 - mean) / std


def _augment(X, rng):
    # SpecAugment-style masking: one random time band and one random frequency band set to the channel mean.
    n, height, width, _ = X.shape
    time_width = rng.integers(0, MAX_TIME_MASK + 1, n)
    time_start = rng.integers(0, width - time_width + 1)
    frequency_width = rng.integers(0, MAX_FREQUENCY_MASK + 1, n)
    frequency_start = rng.integers(0, height - frequency_width + 1)
    columns = np.arange(width)
    rows = np.arange(height)
    keep_time = ~((columns >= time_start[:, None]) & (columns < (time_start + time_width)[:, None]))
    keep_frequency = ~((rows >= frequency_start[:, None]) & (rows < (frequency_start + frequency_width)[:, None]))
    return X * keep_time[:, None, :, None] * keep_frequency[:, :, None, None]


def predict_proba(network, images, mean, std):
    return np.concatenate(
        [
            network.predict_proba(_normalize(images[start:start + EVAL_BATCH_SIZE], mean, std))
            for start in range(0, len(images), EVAL_BATCH_SIZE)
        ]
    ).astype(np.float64)


def _split(y, groups, fraction, indices):
    splitter = StratifiedGroupKFold(n_splits=int(round(1 / fraction)), shuffle=True, random_state=RANDOM_STATE)
    first, second = next(splitter.split(np.zeros(len(indices)), y[indices], groups[indices]))
    return indices[first], indices[second]


def _load_classifier_module():
    spec = importlib.util.spec_from_file_location("cnn_genres_classifier", CLASSIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=EPOCHS):
    rng = np.random.default_rng(RANDOM_STATE)
    images, filenames, labels, song_ids, data_sha256 = load_dataset(data_path)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = encoder.classes_.tolist()

    train_val_idx, test_idx = _split(y, song_ids, HOLDOUT_FRACTION, np.arange(len(y)))
    train_idx, val_idx = _split(y, song_ids, VALIDATION_FRACTION, train_val_idx)

    mean, std = _channel_statistics(images[train_idx])
    network = CNN(len(classes), rng)
    optimizer = AdamW(network.layers, LEARNING_RATE)

    history = []
    best_state = network.state()
    best_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_since_lr_change = 0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        order = rng.permutation(train_idx)
        train_loss = 0.0
        train_correct = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            X = _augment(_normalize(images[batch], mean, std), rng)
            proba = softmax(network.forward(X, training=True, rng=rng))
            targets = y[batch]
            rows = np.arange(len(batch))
            train_loss += float(-np.log(proba[rows, targets] + 1e-12).sum())
            train_correct += int((proba.argmax(axis=1) == targets).sum())
            dlogits = proba
            dlogits[rows, targets] -= 1.0
            network.backward((dlogits / len(batch)).astype(np.float32))
            optimizer.step()

        val_proba = predict_proba(network, images[val_idx], mean, std)
        val_loss = float(log_loss(y[val_idx], val_proba, labels=np.arange(len(classes))))
        val_accuracy = float(accuracy_score(y[val_idx], val_proba.argmax(axis=1)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(order),
                "train_accuracy": train_correct / len(order),
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "learning_rate": optimizer.learning_rate,
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {history[-1]['train_loss']:.4f} acc {history[-1]['train_accuracy']:.4f} | "
            f"validation loss {val_loss:.4f} acc {val_accuracy:.4f} | "
            f"lr {optimizer.learning_rate:.2e} | {history[-1]['seconds']:.0f}s",
            flush=True,
        )

        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, network.state()
            epochs_without_improvement = 0
            epochs_since_lr_change = 0
        else:
            epochs_without_improvement += 1
            epochs_since_lr_change += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping: no validation improvement in {EARLY_STOPPING_PATIENCE} epochs")
                break
            if epochs_since_lr_change >= LR_PATIENCE:
                optimizer.learning_rate *= LR_FACTOR
                epochs_since_lr_change = 0

    network.load_state(best_state)

    test_proba = predict_proba(network, images[test_idx], mean, std)
    y_test = labels[test_idx]
    y_pred = np.asarray(classes, dtype=object)[test_proba.argmax(axis=1)]

    model_input = {
        "source": "data/mel_spectrograms PNG (RGB)",
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "width": INPUT_WIDTH,
        "height": INPUT_HEIGHT,
        "resample": "box",
        "shape": [INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH],
        "layout": "NCHW",
        "scale": "1/255",
        "mean": mean,
        "std": std,
    }
    model_spec = json.loads(
        json.dumps(
            {
                "class": "numpy CNN",
                "classes": classes,
                "input": model_input,
                "layers": network.export(),
            },
            default=_json_default,
        )
    )

    classifier = _load_classifier_module()
    json_layers = [classifier._load_layer(layer) for layer in model_spec["layers"]]
    X_test = np.concatenate(
        [
            classifier.forward(_normalize(images[test_idx[start:start + EVAL_BATCH_SIZE]], mean, std).transpose(0, 3, 1, 2), json_layers)
            for start in range(0, len(test_idx), EVAL_BATCH_SIZE)
        ]
    )
    export_check = {
        "predictions_identical": bool((X_test.argmax(axis=1) == test_proba.argmax(axis=1)).all()),
        "max_probability_difference": float(np.abs(X_test - test_proba).max()),
    }

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
    total = cm.sum()
    tp = np.diag(cm)
    tn = total - cm.sum(axis=0) - cm.sum(axis=1) + tp

    test_songs = song_ids[test_idx]
    unique_songs, song_index = np.unique(test_songs, return_inverse=True)
    song_proba = np.zeros((len(unique_songs), len(classes)))
    np.add.at(song_proba, song_index, test_proba)
    song_true = np.asarray([labels[test_idx][song_index == i][0] for i in range(len(unique_songs))])
    song_pred = np.asarray(classes)[song_proba.argmax(axis=1)]

    layers_summary = [
        {key: layer[key] for key in ("name", "type", "output_shape", "parameters")}
        for layer in model_spec["layers"]
    ]
    best = history[best_epoch - 1]

    results = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "data_path": str(data_path),
            "data_sha256": data_sha256,
            "data_images": len(images),
            "data_songs": int(len(np.unique(song_ids))),
        },
        "config": {
            "random_state": RANDOM_STATE,
            "holdout_fraction": HOLDOUT_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "song_id_pattern": SONG_ID_PATTERN,
            "conv_filters": CONV_FILTERS,
            "kernel_size": KERNEL_SIZE,
            "pool_size": POOL_SIZE,
            "dense_units": DENSE_UNITS,
            "dropout": DROPOUT,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_betas": ADAM_BETAS,
            "lr_factor": LR_FACTOR,
            "lr_patience": LR_PATIENCE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "augmentation": {"max_time_mask": MAX_TIME_MASK, "max_frequency_mask": MAX_FREQUENCY_MASK},
        },
        "split": {
            "train_fragments": len(train_idx),
            "validation_fragments": len(val_idx),
            "test_fragments": len(test_idx),
            "train_songs": sorted(np.unique(song_ids[train_idx]).tolist()),
            "validation_songs": sorted(np.unique(song_ids[val_idx]).tolist()),
            "test_songs": sorted(np.unique(song_ids[test_idx]).tolist()),
        },
        "training": {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_validation_loss": best["validation_loss"],
            "best_validation_accuracy": best["validation_accuracy"],
            "train_accuracy_at_best": best["train_accuracy"],
            "total_seconds": sum(row["seconds"] for row in history),
            "history": history,
        },
        "holdout": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "loss": float(log_loss(y[test_idx], test_proba, labels=np.arange(len(classes)))),
            "song_accuracy": float(accuracy_score(song_true, song_pred)),
            "per_genre": {
                label: {
                    "accuracy": float((tp[i] + tn[i]) / total),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, label in enumerate(classes)
            },
            "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        },
        "export_check": export_check,
        "architecture": {
            "input_layer": {key: model_input[key] for key in ("source", "shape", "layout", "resample", "scale")},
            "hidden_layers": [layer for layer in layers_summary if layer["name"] not in ("output", "softmax")],
            "output_layer": [layer for layer in layers_summary if layer["name"] in ("output", "softmax")],
            "total_parameters": sum(layer["parameters"] for layer in layers_summary),
        },
        "model": model_spec,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(results, f, separators=(",", ":"), ensure_ascii=False, default=_json_default)

    print(f"Images: {len(images)} | Songs: {results['metadata']['data_songs']}")
    print(f"Train fragments: {len(train_idx)} | Validation fragments: {len(val_idx)} | Holdout fragments: {len(test_idx)}")
    print(f"Parameters: {results['architecture']['total_parameters']:,}")
    print(f"Best epoch: {best_epoch} | Validation accuracy: {best['validation_accuracy']:.4f}")
    print(f"Holdout accuracy: {results['holdout']['accuracy']:.4f} | Song accuracy: {results['holdout']['song_accuracy']:.4f}")
    print(f"Export check: {export_check}")
    print(f"Saved: {model_path} ({model_path.stat().st_size / 1024 ** 2:.1f} MB)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(args.data, args.model, args.epochs)
