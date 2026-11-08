import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

DATA_PATH = "data/features_3_sec.csv"
MODEL_PATH = "models/random_forest.json"
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.2
CV_FOLDS = 5
N_ESTIMATORS = 500
SONG_ID_PATTERN = r"([A-Za-z]+)(\d{5})(\d{3})"
NON_FEATURE_COLUMNS = ["filename", "length", "label", "song_id"]
PARAM_GRID = {
    "max_depth": [None, 15, 25],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.33],
}


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


def _leaf_values_are_normalized():
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    return (major, minor) >= (1, 4)


def _export_trees(rf_model):
    normalized = _leaf_values_are_normalized()
    trees = []
    for estimator in rf_model.estimators_:
        tree = estimator.tree_
        is_leaf = tree.children_left == -1
        values = tree.value[:, 0, : len(rf_model.classes_)]
        if not normalized:
            values = values.copy()
            normalizer = values.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            values /= normalizer
        trees.append(
            {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "leaf_values": values[is_leaf].tolist(),
            }
        )
    return trees


class RandomForestFromJSON:
    def __init__(self, model):
        self.classes = list(model["classes"])
        self.feature_names = list(model["feature_names"])
        self.trees = []
        for tree in model["trees"]:
            left = np.asarray(tree["children_left"], dtype=np.int64)
            is_leaf = left == -1
            leaf_index = np.full(left.shape[0], -1, dtype=np.int64)
            leaf_index[is_leaf] = np.arange(is_leaf.sum())
            self.trees.append(
                (
                    left,
                    np.asarray(tree["children_right"], dtype=np.int64),
                    np.asarray(tree["feature"], dtype=np.int64),
                    np.asarray(tree["threshold"], dtype=np.float64),
                    leaf_index,
                    np.asarray(tree["leaf_values"], dtype=np.float64),
                )
            )

    def predict_proba(self, features):
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        X = features[self.feature_names].to_numpy(dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("Features contain NaN or infinite values")
        rows = np.arange(X.shape[0])
        proba = np.zeros((X.shape[0], len(self.classes)), dtype=np.float64)
        for left, right, feature, threshold, leaf_index, leaf_values in self.trees:
            node = np.zeros(X.shape[0], dtype=np.int64)
            active = left[node] != -1
            while active.any():
                current = node[active]
                go_left = X[rows[active], feature[current]] <= threshold[current]
                node[active] = np.where(go_left, left[current], right[current])
                active = left[node] != -1
            proba += leaf_values[leaf_index[node]]
        proba /= len(self.trees)
        return proba

    def predict(self, features):
        return np.asarray(self.classes, dtype=object)[self.predict_proba(features).argmax(axis=1)]

    def __call__(self, features):
        return self.predict(features)


def load_random_forest(model_path=MODEL_PATH):
    with open(model_path, encoding="utf-8") as f:
        return RandomForestFromJSON(json.load(f)["model"])


def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(data_path)
    id_parts = df["filename"].astype(str).str.extract(SONG_ID_PATTERN)
    df["song_id"] = id_parts[0].str.lower() + id_parts[1]
    if df["song_id"].isna().any():
        raise ValueError(f"{df['song_id'].isna().sum()} filenames do not match {SONG_ID_PATTERN}")

    feature_names = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_names]
    y = df["label"]
    groups = df["song_id"]

    holdout_splitter = StratifiedGroupKFold(
        n_splits=int(round(1 / HOLDOUT_FRACTION)),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(holdout_splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)

    search = GridSearchCV(
        estimator=RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1),
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        return_train_score=True,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups_train)

    rf_model = search.best_estimator_
    rf_model.set_params(n_jobs=1)
    labels = rf_model.classes_.tolist()

    model_spec = {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": rf_model.get_params(),
        "classes": labels,
        "feature_names": feature_names,
        "trees": _export_trees(rf_model),
    }

    json_model = RandomForestFromJSON(model_spec)
    json_proba = json_model.predict_proba(X_test)
    y_pred = np.asarray(labels, dtype=object)[json_proba.argmax(axis=1)]
    sklearn_proba = rf_model.predict_proba(X_test)
    export_check = {
        "predictions_identical": bool((rf_model.predict(X_test) == y_pred).all()),
        "probabilities_identical": bool(np.array_equal(sklearn_proba, json_proba)),
    }

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )
    total = cm.sum()
    tp = np.diag(cm)
    tn = total - cm.sum(axis=0) - cm.sum(axis=1) + tp

    cv_res = search.cv_results_
    best = search.best_index_

    with open(data_path, "rb") as f:
        data_sha256 = hashlib.sha256(f.read()).hexdigest()

    results = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "data_path": str(data_path),
            "data_sha256": data_sha256,
            "data_rows": len(df),
            "data_songs": int(groups.nunique()),
        },
        "config": {
            "random_state": RANDOM_STATE,
            "holdout_fraction": HOLDOUT_FRACTION,
            "cv_folds": CV_FOLDS,
            "n_estimators": N_ESTIMATORS,
            "param_grid": PARAM_GRID,
            "song_id_pattern": SONG_ID_PATTERN,
        },
        "split": {
            "train_fragments": len(X_train),
            "test_fragments": len(X_test),
            "train_songs": sorted(groups_train.unique().tolist()),
            "test_songs": sorted(groups_test.unique().tolist()),
        },
        "cross_validation": {
            "best_params": search.best_params_,
            "train_accuracy": float(cv_res["mean_train_score"][best]),
            "validation_accuracy": float(search.best_score_),
            "validation_std": float(cv_res["std_test_score"][best]),
            "grid": sorted(
                [
                    {
                        "rank": int(cv_res["rank_test_score"][i]),
                        "params": cv_res["params"][i],
                        "mean_validation_accuracy": float(cv_res["mean_test_score"][i]),
                        "std_validation_accuracy": float(cv_res["std_test_score"][i]),
                        "mean_train_accuracy": float(cv_res["mean_train_score"][i]),
                    }
                    for i in range(len(cv_res["params"]))
                ],
                key=lambda r: r["rank"],
            ),
        },
        "holdout": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "per_genre": {
                label: {
                    "accuracy": float((tp[i] + tn[i]) / total),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, label in enumerate(labels)
            },
            "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        },
        "export_check": export_check,
        "feature_importances": dict(
            sorted(
                zip(feature_names, rf_model.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
        "model": model_spec,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(results, f, separators=(",", ":"), ensure_ascii=False, default=_json_default)

    print(f"Rows: {len(df)} | Songs: {groups.nunique()}")
    print(f"Train fragments: {len(X_train)} | Holdout fragments: {len(X_test)}")
    print(f"Best params: {search.best_params_}")
    print(f"CV train accuracy: {results['cross_validation']['train_accuracy']:.4f}")
    print(f"CV validation accuracy: {search.best_score_:.4f} ± {results['cross_validation']['validation_std']:.4f}")
    print(f"Holdout accuracy: {results['holdout']['accuracy']:.4f}")
    print(f"Export check: {export_check}")
    print(f"Saved: {model_path} ({model_path.stat().st_size / 1024 ** 2:.1f} MB)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()
    train(args.data, args.model)