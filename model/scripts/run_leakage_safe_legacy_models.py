from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from run_extended_model_experiment import is_rating_derived_feature
from run_full_cv_experiment import FoldSplit, knn_predict
from run_small_portion_experiment import TARGET_COLUMN, load_nodes, mae, mse, rmse, train_graph_mlp


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standardize(train_x: np.ndarray, valid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (valid_x - mean) / std


def metric_row(
    model: str,
    params: dict[str, Any],
    fold: int,
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    return {
        "feature_set": "leakage_safe",
        "model": model,
        "params": ";".join(f"{key}={value}" for key, value in params.items()),
        "fold": fold,
        "mse": mse(y_true, prediction),
        "rmse": rmse(y_true, prediction),
        "mae": mae(y_true, prediction),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["params"]), []).append(row)

    output = []
    for (model, params), values in groups.items():
        output.append(
            {
                "feature_set": "leakage_safe",
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
    return sorted(output, key=lambda row: row["mse_mean"])


def best_per_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["model"] not in best or row["mse_mean"] < best[row["model"]]["mse_mean"]:
            best[row["model"]] = row
    return sorted(best.values(), key=lambda row: row["mse_mean"])


def write_markdown(path: Path, best_rows: list[dict[str, Any]], feature_count: int) -> None:
    lines = [
        "# Leakage-safe KNN and Graph MLP",
        "",
        f"- lectures: 753",
        f"- features: {feature_count}",
        "- validation: 5-fold cross-validation",
        "- excluded: rating-derived sentiment ratios and positive/negative TF-IDF",
        "- Graph MLP evaluation: fixed final epoch prediction; validation-best epoch was not used",
        "",
        "| Rank | Model | Best parameters | MSE | RMSE | MAE |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(best_rows, start=1):
        lines.append(
            f"| {rank} | {row['model']} | `{row['params']}` | "
            f"{row['mse_mean']:.6f} | {row['rmse_mean']:.6f} | {row['mae_mean']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the existing KNN and Graph MLP with leakage-safe features.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/experiments/leakage_safe_legacy"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, all_features = load_nodes(args.nodes)
    features = [column for column in all_features if not is_rating_derived_feature(column)]
    x_all = nodes[features].to_numpy(dtype=float)
    y_all = nodes[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    knn_ks = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50]
    mlp_configs = [
        {"hidden_dim": 8, "learning_rate": 0.003, "epochs": 300},
        {"hidden_dim": 16, "learning_rate": 0.001, "epochs": 300},
        {"hidden_dim": 16, "learning_rate": 0.003, "epochs": 500},
        {"hidden_dim": 32, "learning_rate": 0.003, "epochs": 500},
        {"hidden_dim": 64, "learning_rate": 0.01, "epochs": 800},
    ]

    fold_rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_all), start=1):
        x_train, x_valid = standardize(x_all[train_idx], x_all[valid_idx])
        y_train, y_valid = y_all[train_idx], y_all[valid_idx]
        split = FoldSplit(x_train=x_train, y_train=y_train, x_test=x_valid, y_test=y_valid)

        for k in knn_ks:
            prediction = knn_predict(split, k=k)
            fold_rows.append(metric_row("Content KNN", {"k": k}, fold_idx, y_valid, prediction))

        for config in mlp_configs:
            final_prediction, _validation_best_prediction, _history = train_graph_mlp(
                split=split,  # type: ignore[arg-type]
                hidden_dim=config["hidden_dim"],
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                seed=args.seed + fold_idx,
            )
            fold_rows.append(
                metric_row(
                    "Graph-augmented MLP",
                    config,
                    fold_idx,
                    y_valid,
                    final_prediction,
                )
            )
        print(f"[OK] fold {fold_idx}/{args.folds}")

    all_summaries = summarize(fold_rows)
    best_rows = best_per_model(all_summaries)
    write_csv(args.out_dir / "legacy_fold_metrics.csv", fold_rows)
    write_csv(args.out_dir / "legacy_all_configs.csv", all_summaries)
    write_csv(args.out_dir / "legacy_best_models.csv", best_rows)
    write_markdown(args.out_dir / "legacy_summary.md", best_rows, len(features))

    print(f"[OK] leakage-safe features: {len(features)}")
    for row in best_rows:
        print(row)
    print(f"[OK] output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
