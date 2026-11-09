"""GPU (PyTorch) version of cnn_genres.py.

Same data, splits, architecture, optimizer, schedule and augmentation as the NumPy script,
but the model runs on CUDA with cuDNN and gradients come from autograd.
The exported JSON uses the same format, so cnn_genres_classifier.py
(pure NumPy inference) can load it unchanged.
"""

import argparse
import json
import os
import platform
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch import nn

from cnn_genres import (
    ADAM_BETAS,
    ADAM_EPS,
    BATCH_SIZE,
    BN_EPS,
    BN_MOMENTUM,
    CONV_FILTERS,
    DATA_PATH,
    DENSE_UNITS,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    HOLDOUT_FRACTION,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_CHANNELS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    KERNEL_SIZE,
    LEARNING_RATE,
    LR_FACTOR,
    LR_PATIENCE,
    MAX_FREQUENCY_MASK,
    MAX_TIME_MASK,
    POOL_SIZE,
    RANDOM_STATE,
    SONG_ID_PATTERN,
    VALIDATION_FRACTION,
    WEIGHT_DECAY,
    _channel_statistics,
    _json_default,
    _load_classifier_module,
    _normalize,
    _split,
    load_dataset,
)

MODEL_PATH = "models/cnn.json"
GPU_EVAL_BATCH_SIZE = 256


class TimeMean(nn.Module):
    # NCHW: average over the time axis (width), leaving (batch, channels, frequency).
    def forward(self, X):
        return X.mean(dim=3)


def build_network(n_classes):
    layers = []
    channels, height = INPUT_CHANNELS, INPUT_HEIGHT
    for index, filters in enumerate(CONV_FILTERS, start=1):
        layers += [
            (f"conv{index}", nn.Conv2d(channels, filters, KERNEL_SIZE, padding=KERNEL_SIZE // 2)),
            (f"batch_norm{index}", nn.BatchNorm2d(filters, eps=BN_EPS, momentum=BN_MOMENTUM)),
            (f"relu{index}", nn.ReLU()),
            (f"max_pool{index}", nn.MaxPool2d(POOL_SIZE)),
        ]
        channels, height = filters, height // POOL_SIZE
    layers += [
        ("time_mean", TimeMean()),
        ("flatten", nn.Flatten()),  # flattens (channels, frequency), the order the classifier expects
        ("dense", nn.Linear(channels * height, DENSE_UNITS)),
        ("relu_dense", nn.ReLU()),
        ("dropout", nn.Dropout(DROPOUT)),
        ("output", nn.Linear(DENSE_UNITS, n_classes)),
    ]
    network = nn.Sequential(OrderedDict(layers))
    # He-normal weights and zero biases, as in the NumPy implementation.
    for module in network.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(module.bias)
    return network


def build_optimizer(network):
    # Weight decay only on conv/linear weights, like the NumPy AdamW.
    decay = [p for name, p in network.named_parameters() if name.endswith("weight") and p.ndim > 1]
    no_decay = [p for name, p in network.named_parameters() if not (name.endswith("weight") and p.ndim > 1)]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def export_layers(network, device):
    layers = []
    X = torch.zeros((1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), device=device)
    network.eval()
    with torch.no_grad():
        for name, module in network.named_children():
            X = module(X)
            if isinstance(module, nn.Conv2d):
                spec = {
                    "type": "conv2d",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": list(module.kernel_size),
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                    "weight": _numpy(module.weight),
                    "bias": _numpy(module.bias),
                }
            elif isinstance(module, nn.BatchNorm2d):
                spec = {
                    "type": "batch_norm2d",
                    "num_features": module.num_features,
                    "eps": module.eps,
                    "momentum": module.momentum,
                    "weight": _numpy(module.weight),
                    "bias": _numpy(module.bias),
                    "running_mean": _numpy(module.running_mean),
                    "running_var": _numpy(module.running_var),
                }
            elif isinstance(module, nn.ReLU):
                spec = {"type": "relu"}
            elif isinstance(module, nn.MaxPool2d):
                spec = {"type": "max_pool2d", "kernel_size": [POOL_SIZE, POOL_SIZE], "stride": [POOL_SIZE, POOL_SIZE]}
            elif isinstance(module, TimeMean):
                spec = {"type": "mean", "axis": 3, "dimension": "time"}
            elif isinstance(module, nn.Flatten):
                spec = {"type": "flatten"}
            elif isinstance(module, nn.Linear):
                spec = {
                    "type": "linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "weight": _numpy(module.weight),
                    "bias": _numpy(module.bias),
                }
            elif isinstance(module, nn.Dropout):
                spec = {"type": "dropout", "p": module.p}
            else:
                raise TypeError(f"Cannot export {type(module).__name__}")
            layers.append(
                {
                    "name": name,
                    "role": "output" if name == "output" else "hidden",
                    **spec,
                    "output_shape": list(X.shape[1:]),
                    "parameters": int(sum(p.numel() for p in module.parameters())),
                }
            )
        X = torch.softmax(X, dim=1)
    layers.append({"name": "softmax", "role": "output", "type": "softmax", "output_shape": list(X.shape[1:]), "parameters": 0})
    return layers


def _gpu_normalize(images, mean, std):
    # images: uint8 NCHW on the device; mean/std: (1, C, 1, 1) float tensors.
    return (images.float() / 255.0 - mean) / std


def _augment(X, rng):
    # SpecAugment-style masking: one random time band and one random frequency band set to the channel mean.
    n, _, height, width = X.shape
    time_width = rng.integers(0, MAX_TIME_MASK + 1, n)
    time_start = rng.integers(0, width - time_width + 1)
    frequency_width = rng.integers(0, MAX_FREQUENCY_MASK + 1, n)
    frequency_start = rng.integers(0, height - frequency_width + 1)
    device = X.device
    time_start, time_end = (torch.from_numpy(a).to(device) for a in (time_start, time_start + time_width))
    frequency_start, frequency_end = (torch.from_numpy(a).to(device) for a in (frequency_start, frequency_start + frequency_width))
    columns = torch.arange(width, device=device)
    rows = torch.arange(height, device=device)
    keep_time = ~((columns >= time_start[:, None]) & (columns < time_end[:, None]))
    keep_frequency = ~((rows >= frequency_start[:, None]) & (rows < frequency_end[:, None]))
    return X * keep_time[:, None, None, :] * keep_frequency[:, None, :, None]


@torch.no_grad()
def predict_proba(network, images, indices, mean, std):
    network.eval()
    proba = []
    for start in range(0, len(indices), GPU_EVAL_BATCH_SIZE):
        batch = indices[start:start + GPU_EVAL_BATCH_SIZE]
        logits = network(_gpu_normalize(images[batch], mean, std))
        # Softmax in float64 so rows sum to 1 at the precision sklearn checks.
        proba.append(torch.softmax(logits.double(), dim=1))
    return _numpy(torch.cat(proba))


def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=EPOCHS):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Device: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        device = torch.device("cpu")
        print("Device: CPU (CUDA not available, training will be slow)", flush=True)

    rng = np.random.default_rng(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    images, filenames, labels, song_ids, data_sha256 = load_dataset(data_path)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = encoder.classes_.tolist()

    train_val_idx, test_idx = _split(y, song_ids, HOLDOUT_FRACTION, np.arange(len(y)))
    train_idx, val_idx = _split(y, song_ids, VALIDATION_FRACTION, train_val_idx)

    mean, std = _channel_statistics(images[train_idx])

    # The whole dataset fits on the GPU as uint8 (~1 GB), so batches never leave the device.
    images_gpu = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).contiguous()
    y_gpu = torch.from_numpy(y).to(device)
    mean_gpu = torch.from_numpy(mean).to(device).view(1, -1, 1, 1)
    std_gpu = torch.from_numpy(std).to(device).view(1, -1, 1, 1)
    val_idx_gpu = torch.from_numpy(val_idx).to(device)
    test_idx_gpu = torch.from_numpy(test_idx).to(device)

    network = build_network(len(classes)).to(device)
    optimizer = build_optimizer(network)
    learning_rate = LEARNING_RATE

    history = []
    best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
    best_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_since_lr_change = 0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        network.train()
        order = torch.from_numpy(rng.permutation(train_idx)).to(device)
        train_loss = torch.zeros((), device=device)
        train_correct = torch.zeros((), device=device, dtype=torch.long)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            X = _augment(_gpu_normalize(images_gpu[batch], mean_gpu, std_gpu), rng)
            targets = y_gpu[batch]
            logits = network(X)
            loss_sum = F.cross_entropy(logits, targets, reduction="sum")
            optimizer.zero_grad(set_to_none=True)
            (loss_sum / len(batch)).backward()
            optimizer.step()
            train_loss += loss_sum.detach()
            train_correct += (logits.argmax(dim=1) == targets).sum()

        val_proba = predict_proba(network, images_gpu, val_idx_gpu, mean_gpu, std_gpu)
        val_loss = float(log_loss(y[val_idx], val_proba, labels=np.arange(len(classes))))
        val_accuracy = float(accuracy_score(y[val_idx], val_proba.argmax(axis=1)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss.item() / len(order),
                "train_accuracy": train_correct.item() / len(order),
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "learning_rate": learning_rate,
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {history[-1]['train_loss']:.4f} acc {history[-1]['train_accuracy']:.4f} | "
            f"validation loss {val_loss:.4f} acc {val_accuracy:.4f} | "
            f"lr {learning_rate:.2e} | {history[-1]['seconds']:.1f}s",
            flush=True,
        )

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            epochs_without_improvement = 0
            epochs_since_lr_change = 0
        else:
            epochs_without_improvement += 1
            epochs_since_lr_change += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping: no validation improvement in {EARLY_STOPPING_PATIENCE} epochs")
                break
            if epochs_since_lr_change >= LR_PATIENCE:
                learning_rate *= LR_FACTOR
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                epochs_since_lr_change = 0

    network.load_state_dict(best_state)

    test_proba = predict_proba(network, images_gpu, test_idx_gpu, mean_gpu, std_gpu)
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
                "class": "torch CNN",
                "classes": classes,
                "input": model_input,
                "layers": export_layers(network, device),
            },
            default=_json_default,
        )
    )

    # Verify that the NumPy classifier reproduces the PyTorch predictions from the exported JSON.
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
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "data_path": str(data_path),
            "data_sha256": data_sha256,
            "data_images": len(images),
            "data_songs": int(len(np.unique(song_ids))),
        },
        "config": {
            "framework": "pytorch",
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
    exit_code = 0
    try:
        train(args.data, args.model, args.epochs)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    # PyTorch 2.14 on Python 3.15 / Windows crashes with an access violation while tearing down
    # its autograd threads at interpreter shutdown (any script that called .backward() does this).
    # Everything is already saved here, so skip the normal shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
