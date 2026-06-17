from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from run_extended_model_experiment import is_rating_derived_feature
from run_small_portion_experiment import TARGET_COLUMN, load_nodes, mae, mse, rmse


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def params_text(params: dict[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in sorted(params.items()))


def main() -> int:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    parser = argparse.ArgumentParser(description="Evaluate a standard MLP using the leakage-safe 16 features.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/tfidf16_mlp"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, all_features = load_nodes(args.nodes)
    features = [column for column in all_features if not is_rating_derived_feature(column)]
    x = nodes[features].to_numpy(dtype=float)
    y = nodes[TARGET_COLUMN].to_numpy(dtype=float)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(nodes))

    grid = {
        "hidden_layer_sizes": [(8,), (16,), (32,), (64,), (32, 16), (64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [0.001, 0.01, 0.1],
        "learning_rate_init": [0.001, 0.003],
    }

    summaries: list[dict[str, Any]] = []
    best_fold_rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None

    for params in ParameterGrid(grid):
        fold_rows = []
        for fold_idx, (train_idx, valid_idx) in enumerate(folds, start=1):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_valid = scaler.transform(x[valid_idx])
            model = MLPRegressor(
                solver="adam",
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=args.seed + fold_idx,
                **params,
            )
            model.fit(x_train, y[train_idx])
            prediction = np.clip(model.predict(x_valid), 0.0, 1.0)
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "mse": mse(y[valid_idx], prediction),
                    "rmse": rmse(y[valid_idx], prediction),
                    "mae": mae(y[valid_idx], prediction),
                    "epochs_run": model.n_iter_,
                }
            )

        summary = {
            "feature_set": "tfidf16_leakage_safe",
            "model": "MLP",
            "params": params_text(params),
            "folds": 5,
            "mse_mean": round(float(np.mean([row["mse"] for row in fold_rows])), 8),
            "mse_std": round(float(np.std([row["mse"] for row in fold_rows])), 8),
            "rmse_mean": round(float(np.mean([row["rmse"] for row in fold_rows])), 8),
            "rmse_std": round(float(np.std([row["rmse"] for row in fold_rows])), 8),
            "mae_mean": round(float(np.mean([row["mae"] for row in fold_rows])), 8),
            "mae_std": round(float(np.std([row["mae"] for row in fold_rows])), 8),
            "epochs_mean": round(float(np.mean([row["epochs_run"] for row in fold_rows])), 2),
        }
        summaries.append(summary)
        if best_summary is None or summary["mse_mean"] < best_summary["mse_mean"]:
            best_summary = summary
            best_fold_rows = fold_rows

    assert best_summary is not None
    summaries.sort(key=lambda row: row["mse_mean"])
    write_csv(args.out_dir / "tfidf16_mlp_all_configs.csv", summaries)
    write_csv(args.out_dir / "tfidf16_mlp_best.csv", [best_summary])
    write_csv(
        args.out_dir / "tfidf16_mlp_best_fold_metrics.csv",
        [
            {
                "feature_set": "tfidf16_leakage_safe",
                "model": "MLP",
                "params": best_summary["params"],
                **row,
            }
            for row in best_fold_rows
        ],
    )
    print(f"[BEST] {best_summary}")
    print(f"[OK] features={len(features)}, output={args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
