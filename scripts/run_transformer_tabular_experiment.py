from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler

from run_small_portion_experiment import BASE_FEATURE_COLUMNS, TARGET_COLUMN


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def params_text(params: dict[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in sorted(params.items()))


def fold_features(
    structured: np.ndarray,
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    structured_train = scaler.fit_transform(structured[train_idx])
    structured_valid = scaler.transform(structured[valid_idx])

    embedding_scaler = StandardScaler()
    embedding_train = embedding_scaler.fit_transform(embeddings[train_idx])
    embedding_valid = embedding_scaler.transform(embeddings[valid_idx])
    pca = PCA(n_components=pca_components, random_state=42)
    embedding_train = pca.fit_transform(embedding_train)
    embedding_valid = pca.transform(embedding_valid)
    return (
        np.c_[structured_train, embedding_train],
        np.c_[structured_valid, embedding_valid],
    )


def make_model(model_name: str, params: dict[str, Any], seed: int) -> Any:
    if model_name == "Ridge":
        return Ridge(**params)
    if model_name == "XGBoost":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **params,
        )
    if model_name == "LightGBM":
        return lgb.LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
    raise ValueError(model_name)


def model_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "Ridge": {
            "alpha": [1.0, 10.0, 100.0],
        },
        "XGBoost": {
            "n_estimators": [200, 500],
            "learning_rate": [0.03, 0.05],
            "max_depth": [2, 3],
            "min_child_weight": [3, 8],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [1.0, 10.0],
        },
        "LightGBM": {
            "n_estimators": [200, 500],
            "learning_rate": [0.03, 0.05],
            "num_leaves": [7, 15],
            "max_depth": [3, 5],
            "min_child_samples": [10, 25],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [1.0, 10.0],
        },
    }


def main() -> int:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    )
    parser = argparse.ArgumentParser(description="Evaluate transformer embeddings plus structured features.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/model/lecture_kobert_embeddings.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/transformer_tabular"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    archive = np.load(args.embeddings)
    embedding_ids = archive["lecture_id"].astype(str)
    embedding_matrix = archive["embeddings"].astype(np.float32)
    embedding_by_id = {lecture_id: embedding for lecture_id, embedding in zip(embedding_ids, embedding_matrix)}

    nodes["lecture_id"] = nodes["lecture_id"].astype(str)
    nodes = nodes[nodes["lecture_id"].isin(embedding_by_id)].reset_index(drop=True)
    embeddings = np.stack([embedding_by_id[lecture_id] for lecture_id in nodes["lecture_id"]])
    structured = nodes[BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target = nodes[TARGET_COLUMN].to_numpy(dtype=np.float32)

    cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(cv.split(nodes))
    pca_options = [16, 32, 64]
    all_rows: list[dict[str, Any]] = []

    for pca_components in pca_options:
        cached_folds = [
            (*fold_features(structured, embeddings, train_idx, valid_idx, pca_components), train_idx, valid_idx)
            for train_idx, valid_idx in splits
        ]
        for model_name, grid in model_grids().items():
            best_summary: dict[str, Any] | None = None
            best_fold_rows: list[dict[str, Any]] = []
            for params in ParameterGrid(grid):
                fold_rows = []
                for fold_idx, (x_train, x_valid, train_idx, valid_idx) in enumerate(cached_folds, start=1):
                    model = make_model(model_name, params, args.seed + fold_idx)
                    model.fit(x_train, target[train_idx])
                    prediction = np.clip(model.predict(x_valid), 0.0, 1.0)
                    fold_mse = float(mean_squared_error(target[valid_idx], prediction))
                    fold_rows.append(
                        {
                            "fold": fold_idx,
                            "mse": fold_mse,
                            "rmse": float(np.sqrt(fold_mse)),
                            "mae": float(mean_absolute_error(target[valid_idx], prediction)),
                        }
                    )

                summary = {
                    "embedding": str(archive["model"][0]),
                    "feature_set": f"structured10+embedding_pca{pca_components}",
                    "model": model_name,
                    "params": params_text(params),
                    "folds": args.folds,
                    "mse_mean": round(float(np.mean([row["mse"] for row in fold_rows])), 8),
                    "mse_std": round(float(np.std([row["mse"] for row in fold_rows])), 8),
                    "rmse_mean": round(float(np.mean([row["rmse"] for row in fold_rows])), 8),
                    "rmse_std": round(float(np.std([row["rmse"] for row in fold_rows])), 8),
                    "mae_mean": round(float(np.mean([row["mae"] for row in fold_rows])), 8),
                    "mae_std": round(float(np.std([row["mae"] for row in fold_rows])), 8),
                }
                if best_summary is None or summary["mse_mean"] < best_summary["mse_mean"]:
                    best_summary = summary
                    best_fold_rows = fold_rows

            assert best_summary is not None
            all_rows.append(best_summary)
            print(
                f"[BEST] {best_summary['feature_set']} {model_name}: "
                f"RMSE={best_summary['rmse_mean']:.6f}, MAE={best_summary['mae_mean']:.6f}"
            )

    all_rows.sort(key=lambda row: row["mse_mean"])
    write_csv(args.out_dir / "transformer_tabular_cv_summary.csv", all_rows)
    print(f"[OK] nodes={len(nodes)}, embedding_dim={embeddings.shape[1]}")
    print(f"[OK] output={args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
