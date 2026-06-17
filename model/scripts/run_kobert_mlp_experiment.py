from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neural_network import MLPRegressor

from run_kobert_all_models import load_data, prepare_folds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def params_text(params: dict[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in sorted(params.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MLP on structured 10 + KoBERT PCA32 features.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/model/lecture_kobert_embeddings.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/kobert_mlp"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, structured, embeddings, target = load_data(args.nodes, args.embeddings)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(nodes))
    folds = prepare_folds(structured, embeddings, splits, pca_components=32)

    grid = {
        "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
        "alpha": [0.001, 0.01, 0.1],
        "learning_rate_init": [0.001, 0.003],
        "activation": ["relu", "tanh"],
    }

    summaries: list[dict[str, Any]] = []
    fold_output: list[dict[str, Any]] = []
    for params in ParameterGrid(grid):
        metrics = []
        for fold_idx, (x_train, x_valid, train_idx, valid_idx) in enumerate(folds, start=1):
            model = MLPRegressor(
                solver="adam",
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=args.seed + fold_idx,
                **params,
            )
            model.fit(x_train, target[train_idx])
            prediction = np.clip(model.predict(x_valid), 0.0, 1.0)
            fold_mse = float(mean_squared_error(target[valid_idx], prediction))
            metrics.append(
                {
                    "fold": fold_idx,
                    "mse": fold_mse,
                    "rmse": float(np.sqrt(fold_mse)),
                    "mae": float(mean_absolute_error(target[valid_idx], prediction)),
                    "epochs_run": model.n_iter_,
                }
            )

        summary = {
            "feature_set": "structured10+kobert_pca32",
            "model": "MLP",
            "params": params_text(params),
            "folds": 5,
            "mse_mean": round(float(np.mean([row["mse"] for row in metrics])), 8),
            "mse_std": round(float(np.std([row["mse"] for row in metrics])), 8),
            "rmse_mean": round(float(np.mean([row["rmse"] for row in metrics])), 8),
            "rmse_std": round(float(np.std([row["rmse"] for row in metrics])), 8),
            "mae_mean": round(float(np.mean([row["mae"] for row in metrics])), 8),
            "mae_std": round(float(np.std([row["mae"] for row in metrics])), 8),
            "epochs_mean": round(float(np.mean([row["epochs_run"] for row in metrics])), 2),
        }
        summaries.append(summary)
        for row in metrics:
            fold_output.append(
                {
                    "feature_set": summary["feature_set"],
                    "model": "MLP",
                    "params": summary["params"],
                    **row,
                }
            )

    summaries.sort(key=lambda row: row["mse_mean"])
    best = summaries[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "kobert_mlp_all_configs.csv", summaries)
    write_csv(args.out_dir / "kobert_mlp_fold_metrics.csv", fold_output)
    write_csv(args.out_dir / "kobert_mlp_best.csv", [best])
    print(f"[BEST] {best}")
    print(f"[OK] output={args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
