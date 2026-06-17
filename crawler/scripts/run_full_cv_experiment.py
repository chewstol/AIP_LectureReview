from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_small_portion_experiment import (
    TARGET_COLUMN,
    load_nodes,
    mae,
    mse,
    rmse,
    set_seed,
    train_graph_mlp,
)


def make_folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = set_seed(seed)
    indices = rng.permutation(n)
    return [fold for fold in np.array_split(indices, k) if len(fold)]


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


class FoldSplit:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


def ridge_predict(split: FoldSplit, alpha: float) -> np.ndarray:
    x_train = np.c_[np.ones((len(split.x_train), 1)), split.x_train]
    x_test = np.c_[np.ones((len(split.x_test), 1)), split.x_test]
    identity = np.eye(x_train.shape[1])
    identity[0, 0] = 0.0
    weights = np.linalg.pinv(x_train.T @ x_train + alpha * identity) @ x_train.T @ split.y_train
    return np.clip(x_test @ weights, 0.0, 1.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True).T
    return (a @ b.T) / np.maximum(a_norm @ b_norm, 1e-12)


def knn_predict(split: FoldSplit, k: int) -> np.ndarray:
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


def mlp_predict(split: FoldSplit, hidden_dim: int, lr: float, epochs: int, seed: int) -> np.ndarray:
    # Reuse the existing training function by duck-typing the fields it needs.
    pred, _best_pred, _history = train_graph_mlp(
        split=split,  # type: ignore[arg-type]
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=lr,
        seed=seed,
    )
    return pred


def metric_row(model: str, params: dict[str, Any], fold: int, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "model": model,
        "params": ";".join(f"{key}={value}" for key, value in params.items()),
        "fold": fold,
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["params"]), []).append(row)

    summary = []
    for (model, params), values in groups.items():
        summary.append(
            {
                "model": model,
                "params": params,
                "folds": len(values),
                "mse_mean": round(float(np.mean([row["mse"] for row in values])), 8),
                "mse_std": round(float(np.std([row["mse"] for row in values])), 8),
                "rmse_mean": round(float(np.mean([row["rmse"] for row in values])), 8),
                "rmse_std": round(float(np.std([row["rmse"] for row in values])), 8),
                "mae_mean": round(float(np.mean([row["mae"] for row in values])), 8),
                "mae_std": round(float(np.std([row["mae"] for row in values])), 8),
            }
        )
    return sorted(summary, key=lambda row: row["mse_mean"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_qualitative_cases(nodes: pd.DataFrame, feature_columns: list[str], out_dir: Path) -> None:
    scenarios = [
        (
            "low_workload",
            {
                "assignment_low_score": 1.0,
                "teamwork_low_score": 1.0,
                "grading_generous_score": 1.0,
                "attendance_light_score": 0.8,
                "exam_light_score": 1.0,
            },
        ),
        (
            "learning_quality",
            {
                "text_teaching_positive_tfidf": 1.0,
                "text_teaching_tfidf": 0.8,
                "grading_generous_score": 0.4,
                "exam_heavy_score": 0.2,
            },
        ),
        (
            "avoid_team_project",
            {
                "teamwork_low_score": 1.0,
                "teamwork_high_score": 0.0,
                "assignment_low_score": 0.8,
                "exam_light_score": 0.7,
                "text_teamwork_negative_tfidf": 0.0,
            },
        ),
    ]

    feature_matrix = nodes[feature_columns].to_numpy(dtype=float)
    rows = []
    for scenario_name, pref in scenarios:
        preference = np.array([pref.get(column, 0.0) for column in feature_columns], dtype=float).reshape(1, -1)
        similarities = cosine_similarity(preference, feature_matrix).reshape(-1)
        ranking = nodes.copy()
        ranking["scenario"] = scenario_name
        ranking["similarity"] = similarities
        ranking["rank_score"] = 0.7 * similarities + 0.3 * ranking[TARGET_COLUMN].astype(float)
        top = ranking.sort_values("rank_score", ascending=False).head(10)
        rows.extend(top[["scenario", "lecture_id", "rank_score", "similarity", TARGET_COLUMN, *feature_columns[:10]]].to_dict("records"))
    write_csv(out_dir / "qualitative_top10_scenarios.csv", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full-data cross-validation experiments.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/full_cv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, feature_columns = load_nodes(args.nodes)
    x_all = nodes[feature_columns].to_numpy(dtype=float)
    y_all = nodes[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)
    folds = make_folds(len(nodes), args.folds, args.seed)

    all_rows: list[dict[str, Any]] = []
    ridge_alphas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    knn_ks = [1, 3, 5, 7, 10, 15, 20, 30]
    mlp_params = [
        {"hidden_dim": 16, "lr": 0.01, "epochs": 300},
        {"hidden_dim": 32, "lr": 0.01, "epochs": 300},
        {"hidden_dim": 64, "lr": 0.03, "epochs": 800},
    ]

    for fold_idx, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(nodes)), test_idx)
        x_train, x_test = standardize(x_all[train_idx], x_all[test_idx])
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        split = FoldSplit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

        mean_pred = np.full_like(y_test, float(y_train.mean()))
        all_rows.append(metric_row("Train Mean", {}, fold_idx, y_test, mean_pred))

        for alpha in ridge_alphas:
            pred = ridge_predict(split, alpha)
            all_rows.append(metric_row("Ridge Regression", {"alpha": alpha}, fold_idx, y_test, pred))

        for k in knn_ks:
            pred = knn_predict(split, k)
            all_rows.append(metric_row("Content KNN", {"k": k}, fold_idx, y_test, pred))

        for params in mlp_params:
            pred = mlp_predict(split, seed=args.seed + fold_idx, **params)
            all_rows.append(metric_row("Graph-augmented MLP", params, fold_idx, y_test, pred))

    summary_rows = summarize(all_rows)
    write_csv(args.out_dir / "cv_fold_metrics.csv", all_rows)
    write_csv(args.out_dir / "cv_summary.csv", summary_rows)
    build_qualitative_cases(nodes, feature_columns, args.out_dir)

    config_rows = [
        {"item": "nodes", "value": len(nodes)},
        {"item": "folds", "value": args.folds},
        {"item": "feature_count", "value": len(feature_columns)},
        {"item": "features", "value": ", ".join(feature_columns)},
        {"item": "target", "value": TARGET_COLUMN},
    ]
    write_csv(args.out_dir / "cv_experiment_config.csv", config_rows)

    print(f"[OK] nodes: {len(nodes)}")
    print(f"[OK] folds: {args.folds}")
    print(f"[OK] feature count: {len(feature_columns)}")
    print("[OK] top CV results:")
    for row in summary_rows[:8]:
        print(row)
    print(f"[OK] output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
