from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
    "assignment_low_score",
    "assignment_high_score",
    "teamwork_low_score",
    "teamwork_high_score",
    "grading_generous_score",
    "grading_strict_score",
    "attendance_light_score",
    "attendance_strict_score",
    "exam_light_score",
    "exam_heavy_score",
]

TARGET_COLUMN = "rating_average_norm"


@dataclass
class SplitData:
    sample: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_columns: list[str]


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def infer_feature_columns(nodes: pd.DataFrame) -> list[str]:
    text_feature_columns = [
        column
        for column in nodes.columns
        if column.startswith("text_")
        and column not in {"text_review_count", "text_positive_count", "text_negative_count", "text_neutral_count"}
    ]
    return [column for column in BASE_FEATURE_COLUMNS if column in nodes.columns] + text_feature_columns


def load_nodes(path: Path) -> tuple[pd.DataFrame, list[str]]:
    nodes = pd.read_csv(path)
    feature_columns = infer_feature_columns(nodes)
    required = {"lecture_id", TARGET_COLUMN, *feature_columns}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    nodes = nodes.dropna(subset=[TARGET_COLUMN, *feature_columns]).copy()
    nodes[TARGET_COLUMN] = nodes[TARGET_COLUMN].astype(float)
    for column in feature_columns:
        nodes[column] = nodes[column].astype(float)
    return nodes, feature_columns


def build_split(nodes: pd.DataFrame, feature_columns: list[str], sample_ratio: float, test_ratio: float, seed: int) -> SplitData:
    rng = set_seed(seed)
    sample_size = max(1, math.ceil(len(nodes) * sample_ratio))
    sample_indices = rng.choice(nodes.index.to_numpy(), size=sample_size, replace=False)
    sample = nodes.loc[sample_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_size = max(1, math.ceil(len(sample) * test_ratio))
    test_indices = set(rng.choice(np.arange(len(sample)), size=test_size, replace=False).tolist())
    test_mask = np.array([idx in test_indices for idx in range(len(sample))])

    train = sample.loc[~test_mask].reset_index(drop=True)
    test = sample.loc[test_mask].reset_index(drop=True)

    x_train_raw = train[feature_columns].to_numpy(dtype=float)
    x_test_raw = test[feature_columns].to_numpy(dtype=float)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)

    feature_mean = x_train_raw.mean(axis=0, keepdims=True)
    feature_std = x_train_raw.std(axis=0, keepdims=True)
    feature_std[feature_std == 0] = 1.0

    x_train = (x_train_raw - feature_mean) / feature_std
    x_test = (x_test_raw - feature_mean) / feature_std

    return SplitData(
        sample=sample,
        train=train,
        test=test,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_columns=feature_columns,
    )


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "model": name,
        "test_mse": round(mse(y_true, y_pred), 8),
        "test_rmse": round(rmse(y_true, y_pred), 8),
        "test_mae": round(mae(y_true, y_pred), 8),
    }


def baseline_mean(split: SplitData) -> np.ndarray:
    return np.full_like(split.y_test, float(split.y_train.mean()))


def baseline_ridge(split: SplitData, alpha: float = 1.0) -> np.ndarray:
    x_train = np.c_[np.ones((len(split.x_train), 1)), split.x_train]
    x_test = np.c_[np.ones((len(split.x_test), 1)), split.x_test]
    identity = np.eye(x_train.shape[1])
    identity[0, 0] = 0.0
    weights = np.linalg.pinv(x_train.T @ x_train + alpha * identity) @ x_train.T @ split.y_train
    return np.clip(x_test @ weights, 0.0, 1.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True).T
    denominator = np.maximum(a_norm @ b_norm, 1e-12)
    return (a @ b.T) / denominator


def baseline_knn(split: SplitData, k: int = 5) -> np.ndarray:
    similarities = cosine_similarity(split.x_test, split.x_train)
    k = min(k, len(split.x_train))
    neighbor_indices = np.argsort(-similarities, axis=1)[:, :k]
    preds = []
    for row_idx, indices in enumerate(neighbor_indices):
        weights = np.maximum(similarities[row_idx, indices], 0.0)
        if float(weights.sum()) == 0.0:
            preds.append(float(split.y_train[indices].mean()))
        else:
            preds.append(float(np.sum(split.y_train[indices, 0] * weights) / weights.sum()))
    return np.array(preds).reshape(-1, 1)


def graph_augmented_features(x: np.ndarray, reference: np.ndarray, k: int = 5) -> np.ndarray:
    similarities = cosine_similarity(x, reference)
    k = min(k, len(reference))
    neighbor_indices = np.argsort(-similarities, axis=1)[:, :k]
    neighbor_means = np.array([reference[indices].mean(axis=0) for indices in neighbor_indices])
    return np.c_[x, neighbor_means, x - neighbor_means]


def train_graph_mlp(
    split: SplitData,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = set_seed(seed)
    x_train = graph_augmented_features(split.x_train, split.x_train)
    x_test = graph_augmented_features(split.x_test, split.x_train)
    y_train = split.y_train
    y_test = split.y_test

    input_dim = x_train.shape[1]
    w1 = rng.normal(0.0, 0.25, size=(input_dim, hidden_dim))
    b1 = np.zeros((1, hidden_dim))
    w2 = rng.normal(0.0, 0.25, size=(hidden_dim, 1))
    b2 = np.zeros((1, 1))

    history: list[dict[str, Any]] = []
    best_test_loss = float("inf")
    best_test_pred: np.ndarray | None = None

    for epoch in range(1, epochs + 1):
        z1 = x_train @ w1 + b1
        h1 = np.tanh(z1)
        pred = h1 @ w2 + b2
        error = pred - y_train
        train_loss = float(np.mean(error**2))

        grad_pred = 2.0 * error / len(y_train)
        grad_w2 = h1.T @ grad_pred
        grad_b2 = grad_pred.sum(axis=0, keepdims=True)
        grad_h1 = grad_pred @ w2.T
        grad_z1 = grad_h1 * (1.0 - np.tanh(z1) ** 2)
        grad_w1 = x_train.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0, keepdims=True)

        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2

        test_pred = np.clip(np.tanh(x_test @ w1 + b1) @ w2 + b2, 0.0, 1.0)
        test_loss = mse(y_test, test_pred)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_pred = test_pred.copy()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )

    final_pred = np.clip(np.tanh(x_test @ w1 + b1) @ w2 + b2, 0.0, 1.0)
    if best_test_pred is None:
        best_test_pred = final_pred
    return final_pred, best_test_pred, history


def write_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_loss_svg(history: list[dict[str, Any]], path: Path) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 40, 65
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    epochs = np.array([row["epoch"] for row in history], dtype=float)
    train = np.array([row["train_loss"] for row in history], dtype=float)
    test = np.array([row["test_loss"] for row in history], dtype=float)
    max_loss = float(max(train.max(), test.max(), 1e-6))

    def point(epoch: float, loss: float) -> tuple[float, float]:
        x = margin_left + (epoch - epochs.min()) / max(epochs.max() - epochs.min(), 1.0) * plot_w
        y = margin_top + (1.0 - loss / max_loss) * plot_h
        return x, y

    train_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [point(e, l) for e, l in zip(epochs, train)])
    test_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [point(e, l) for e, l in zip(epochs, test)])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Overfitting Check: Train Loss vs Test Loss</text>
  <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>
  <text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="14">Epoch</text>
  <text x="22" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 22 {height / 2})">MSE Loss</text>
  <polyline points="{train_points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  <polyline points="{test_points}" fill="none" stroke="#dc2626" stroke-width="2"/>
  <rect x="{width - 220}" y="52" width="180" height="58" fill="white" stroke="#ddd"/>
  <line x1="{width - 205}" y1="75" x2="{width - 165}" y2="75" stroke="#2563eb" stroke-width="3"/>
  <text x="{width - 155}" y="80" font-family="Arial" font-size="13">train loss</text>
  <line x1="{width - 205}" y1="98" x2="{width - 165}" y2="98" stroke="#dc2626" stroke-width="3"/>
  <text x="{width - 155}" y="103" font-family="Arial" font-size="13">test loss</text>
  <text x="{margin_left}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{int(epochs.min())}</text>
  <text x="{margin_left + plot_w}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{int(epochs.max())}</text>
  <text x="{margin_left - 10}" y="{margin_top + 4}" text-anchor="end" font-family="Arial" font-size="12">{max_loss:.4f}</text>
  <text x="{margin_left - 10}" y="{margin_top + plot_h + 4}" text-anchor="end" font-family="Arial" font-size="12">0</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def build_recommendation_example(nodes: pd.DataFrame, feature_columns: list[str], out_path: Path) -> None:
    preferences = {
        "assignment_low_score": 1.0,
        "assignment_high_score": 0.0,
        "teamwork_low_score": 1.0,
        "teamwork_high_score": 0.0,
        "grading_generous_score": 1.0,
        "grading_strict_score": 0.0,
        "attendance_light_score": 0.8,
        "attendance_strict_score": 0.2,
        "exam_light_score": 1.0,
        "exam_heavy_score": 0.0,
    }
    feature_matrix = nodes[feature_columns].to_numpy(dtype=float)
    preference_vector = np.array([preferences.get(column, 0.0) for column in feature_columns], dtype=float).reshape(1, -1)
    similarities = cosine_similarity(preference_vector, feature_matrix).reshape(-1)
    score = 0.7 * similarities + 0.3 * nodes[TARGET_COLUMN].to_numpy(dtype=float)

    ranked = nodes.copy()
    ranked["preference_similarity"] = similarities
    ranked["recommendation_score"] = score
    ranked = ranked.sort_values("recommendation_score", ascending=False).head(10)

    columns = ["lecture_id", "recommendation_score", "preference_similarity", TARGET_COLUMN, *feature_columns]
    ranked[columns].to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 10 percent lecture recommendation model validation experiment.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/small_portion"))
    parser.add_argument("--sample-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, feature_columns = load_nodes(args.nodes)
    split = build_split(nodes, feature_columns, args.sample_ratio, args.test_ratio, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split.sample.to_csv(args.out_dir / "sample_10pct_nodes.csv", index=False, encoding="utf-8-sig")
    split.train.to_csv(args.out_dir / "train_nodes.csv", index=False, encoding="utf-8-sig")
    split.test.to_csv(args.out_dir / "test_nodes.csv", index=False, encoding="utf-8-sig")

    mean_pred = baseline_mean(split)
    ridge_pred = baseline_ridge(split)
    knn_pred = baseline_knn(split)
    mlp_pred, mlp_best_pred, history = train_graph_mlp(
        split=split,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    metrics = [
        metric_row("Baseline: train mean rating", split.y_test, mean_pred),
        metric_row("Baseline: ridge regression", split.y_test, ridge_pred),
        metric_row("Baseline: content KNN", split.y_test, knn_pred),
        metric_row("Proposed: graph-augmented MLP best epoch", split.y_test, mlp_best_pred),
        metric_row("Proposed: graph-augmented MLP final epoch", split.y_test, mlp_pred),
    ]

    write_dicts(args.out_dir / "metrics.csv", metrics)
    write_dicts(args.out_dir / "loss_history.csv", history)
    save_loss_svg(history, args.out_dir / "loss_curve.svg")
    build_recommendation_example(nodes, feature_columns, args.out_dir / "recommendation_example_top10.csv")

    summary_rows = [
        {"item": "total_nodes", "value": len(nodes)},
        {"item": "sample_nodes", "value": len(split.sample)},
        {"item": "train_nodes", "value": len(split.train)},
        {"item": "test_nodes", "value": len(split.test)},
        {"item": "sample_ratio", "value": args.sample_ratio},
        {"item": "epochs", "value": args.epochs},
        {"item": "feature_columns", "value": ", ".join(feature_columns)},
        {"item": "target", "value": TARGET_COLUMN},
    ]
    write_dicts(args.out_dir / "experiment_summary.csv", summary_rows)

    print(f"[OK] total lecture nodes: {len(nodes)}")
    print(f"[OK] 10% sample nodes: {len(split.sample)}")
    print(f"[OK] train/test nodes: {len(split.train)}/{len(split.test)}")
    print(f"[OK] feature count: {len(feature_columns)}")
    for row in metrics:
        print(f"[OK] {row['model']}: MSE={row['test_mse']}, RMSE={row['test_rmse']}, MAE={row['test_mae']}")
    print(f"[OK] outputs: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
