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
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from run_small_portion_experiment import BASE_FEATURE_COLUMNS, TARGET_COLUMN


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def params_text(params: dict[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in sorted(params.items()))


def load_data(nodes_path: Path, embeddings_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    nodes = pd.read_csv(nodes_path)
    nodes["lecture_id"] = nodes["lecture_id"].astype(str)
    archive = np.load(embeddings_path)
    embedding_by_id = dict(zip(archive["lecture_id"].astype(str), archive["embeddings"]))
    nodes = nodes[nodes["lecture_id"].isin(embedding_by_id)].reset_index(drop=True)
    structured = nodes[BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    embeddings = np.stack([embedding_by_id[value] for value in nodes["lecture_id"]]).astype(np.float32)
    target = nodes[TARGET_COLUMN].to_numpy(dtype=np.float32)
    return nodes, structured, embeddings, target


def prepare_folds(
    structured: np.ndarray,
    embeddings: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    pca_components: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    prepared = []
    for train_idx, valid_idx in splits:
        structured_scaler = StandardScaler()
        train_structured = structured_scaler.fit_transform(structured[train_idx])
        valid_structured = structured_scaler.transform(structured[valid_idx])

        embedding_scaler = StandardScaler()
        train_embedding = embedding_scaler.fit_transform(embeddings[train_idx])
        valid_embedding = embedding_scaler.transform(embeddings[valid_idx])
        pca = PCA(n_components=pca_components, random_state=42)
        train_embedding = pca.fit_transform(train_embedding)
        valid_embedding = pca.transform(valid_embedding)

        prepared.append(
            (
                np.c_[train_structured, train_embedding],
                np.c_[valid_structured, valid_embedding],
                train_idx,
                valid_idx,
            )
        )
    return prepared


def model_specs(seed: int) -> list[tuple[str, Any, dict[str, list[Any]]]]:
    return [
        ("Ridge", Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]}),
        (
            "Kernel Ridge-RBF",
            KernelRidge(kernel="rbf"),
            {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "gamma": [0.0001, 0.001, 0.01, 0.1],
            },
        ),
        (
            "Content KNN",
            KNeighborsRegressor(metric="cosine", weights="distance", algorithm="brute"),
            {"n_neighbors": [5, 10, 15, 20, 30, 40]},
        ),
        (
            "SVR-RBF",
            SVR(kernel="rbf"),
            {
                "C": [0.1, 0.3, 1.0, 3.0],
                "epsilon": [0.01, 0.03, 0.05],
                "gamma": ["scale", 0.001, 0.01],
            },
        ),
        (
            "Random Forest",
            RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1),
            {
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 3, 6],
                "max_features": [0.7, 1.0],
            },
        ),
        (
            "Extra Trees",
            ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=1),
            {
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 3, 6],
                "max_features": [0.7, 1.0],
            },
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=seed),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.05],
                "max_depth": [1, 2],
                "loss": ["squared_error", "huber"],
            },
        ),
        (
            "Histogram Gradient Boosting",
            HistGradientBoostingRegressor(random_state=seed),
            {
                "learning_rate": [0.03, 0.05, 0.1],
                "max_iter": [100, 300],
                "max_leaf_nodes": [7, 15],
                "l2_regularization": [0.0, 1.0],
            },
        ),
        (
            "XGBoost",
            xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            ),
            {
                "n_estimators": [200, 500],
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
                "min_child_weight": [3, 8],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "reg_lambda": [1.0, 10.0],
            },
        ),
        (
            "LightGBM",
            lgb.LGBMRegressor(
                objective="regression",
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            ),
            {
                "n_estimators": [200, 500],
                "learning_rate": [0.03, 0.05],
                "num_leaves": [7, 15],
                "max_depth": [3, 5],
                "min_child_samples": [10, 25],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "reg_lambda": [1.0, 10.0],
            },
        ),
    ]


def evaluate(
    estimator: Any,
    params: dict[str, Any],
    prepared_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    target: np.ndarray,
) -> list[dict[str, float]]:
    rows = []
    for fold_idx, (x_train, x_valid, train_idx, valid_idx) in enumerate(prepared_folds, start=1):
        model = estimator.set_params(**params)
        model.fit(x_train, target[train_idx])
        prediction = np.clip(model.predict(x_valid), 0.0, 1.0)
        fold_mse = float(mean_squared_error(target[valid_idx], prediction))
        rows.append(
            {
                "fold": fold_idx,
                "mse": fold_mse,
                "rmse": float(np.sqrt(fold_mse)),
                "mae": float(mean_absolute_error(target[valid_idx], prediction)),
            }
        )
    return rows


def summarize(model_name: str, params: dict[str, Any], rows: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "feature_set": "structured10+kobert_pca32",
        "model": model_name,
        "params": params_text(params),
        "folds": len(rows),
        "mse_mean": round(float(np.mean([row["mse"] for row in rows])), 8),
        "mse_std": round(float(np.std([row["mse"] for row in rows])), 8),
        "rmse_mean": round(float(np.mean([row["rmse"] for row in rows])), 8),
        "rmse_std": round(float(np.std([row["rmse"] for row in rows])), 8),
        "mae_mean": round(float(np.mean([row["mae"] for row in rows])), 8),
        "mae_std": round(float(np.std([row["mae"] for row in rows])), 8),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# KoBERT PCA32 + Structured 10: All Models",
        "",
        "- lectures: 753",
        "- input: structured features 10 + KoBERT PCA components 32",
        "- PCA/scaling: fitted within each training fold",
        "- validation: 5-fold cross-validation",
        "",
        "| Rank | Model | Best parameters | MSE | RMSE | MAE |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['model']} | `{row['params']}` | "
            f"{row['mse_mean']:.6f} | {row['rmse_mean']:.6f} | {row['mae_mean']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    )
    parser = argparse.ArgumentParser(description="Compare all regression models on KoBERT and structured features.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/model/lecture_kobert_embeddings.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/kobert_all_models"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, structured, embeddings, target = load_data(args.nodes, args.embeddings)
    splits = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(nodes))
    prepared_folds = prepare_folds(structured, embeddings, splits, pca_components=32)

    summaries = []
    fold_output = []
    for model_name, estimator, grid in model_specs(args.seed):
        best_summary = None
        best_fold_rows = []
        for params in ParameterGrid(grid):
            fold_rows = evaluate(estimator, params, prepared_folds, target)
            summary = summarize(model_name, params, fold_rows)
            if best_summary is None or summary["mse_mean"] < best_summary["mse_mean"]:
                best_summary = summary
                best_fold_rows = fold_rows
        assert best_summary is not None
        summaries.append(best_summary)
        for row in best_fold_rows:
            fold_output.append(
                {
                    "feature_set": best_summary["feature_set"],
                    "model": model_name,
                    "params": best_summary["params"],
                    **row,
                }
            )
        print(
            f"[BEST] {model_name}: MSE={best_summary['mse_mean']:.6f}, "
            f"RMSE={best_summary['rmse_mean']:.6f}, MAE={best_summary['mae_mean']:.6f}"
        )

    summaries.sort(key=lambda row: row["mse_mean"])
    write_csv(args.out_dir / "kobert_all_models_summary.csv", summaries)
    write_csv(args.out_dir / "kobert_all_models_fold_metrics.csv", fold_output)
    write_markdown(args.out_dir / "kobert_all_models_summary.md", summaries)
    print(f"[OK] output={args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
