from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from run_small_portion_experiment import TARGET_COLUMN, load_nodes


LEAKAGE_DERIVED_COLUMNS = {
    "text_positive_ratio",
    "text_negative_ratio",
    "text_neutral_ratio",
}


def is_rating_derived_feature(column: str) -> bool:
    return (
        column in LEAKAGE_DERIVED_COLUMNS
        or "_positive_tfidf" in column
        or "_negative_tfidf" in column
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_specs(seed: int) -> list[tuple[str, Any, dict[str, list[Any]]]]:
    return [
        (
            "Ridge",
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": [1.0, 10.0, 100.0]},
        ),
        (
            "SVR-RBF",
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            {
                "svr__C": [0.3, 1.0, 3.0, 10.0],
                "svr__epsilon": [0.01, 0.03, 0.05],
                "svr__gamma": ["scale"],
            },
        ),
        (
            "Kernel Ridge-RBF",
            make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
            {
                "kernelridge__alpha": [0.01, 0.1, 1.0],
                "kernelridge__gamma": [0.001, 0.01, 0.1],
            },
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=500,
                random_state=seed,
                n_jobs=-1,
                max_features=1.0,
            ),
            {
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 3, 6],
            },
        ),
        (
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=500,
                random_state=seed,
                n_jobs=-1,
                max_features=1.0,
            ),
            {
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 3, 6],
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
    ]


def param_text(params: dict[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in sorted(params.items()))


def evaluate_configuration(
    estimator: Any,
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    folds: KFold,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(x), start=1):
        model = clone(estimator).set_params(**params)
        model.fit(x[train_idx], y[train_idx])
        prediction = np.clip(model.predict(x[valid_idx]), 0.0, 1.0)
        mse = float(mean_squared_error(y[valid_idx], prediction))
        rows.append(
            {
                "fold": float(fold_idx),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y[valid_idx], prediction)),
            }
        )
    return rows


def summarize_fold_rows(
    feature_set: str,
    model_name: str,
    params: dict[str, Any],
    fold_rows: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "feature_set": feature_set,
        "model": model_name,
        "params": param_text(params),
        "folds": len(fold_rows),
        "mse_mean": round(float(np.mean([row["mse"] for row in fold_rows])), 8),
        "mse_std": round(float(np.std([row["mse"] for row in fold_rows])), 8),
        "rmse_mean": round(float(np.mean([row["rmse"] for row in fold_rows])), 8),
        "rmse_std": round(float(np.std([row["rmse"] for row in fold_rows])), 8),
        "mae_mean": round(float(np.mean([row["mae"] for row in fold_rows])), 8),
        "mae_std": round(float(np.std([row["mae"] for row in fold_rows])), 8),
    }


def build_stacking_model(seed: int) -> StackingRegressor:
    estimators = [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        (
            "svr",
            make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=1.0, epsilon=0.03, gamma="scale"),
            ),
        ),
        (
            "extra",
            ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=10.0),
        cv=5,
        n_jobs=1,
    )


def write_summary_markdown(path: Path, summaries: list[dict[str, Any]], feature_counts: dict[str, int]) -> None:
    lines = [
        "# Extended Model Experiment",
        "",
        "전체 753개 강의에 대해 동일한 5-fold cross-validation으로 표형 데이터 모델을 비교했습니다.",
        "",
        "## Feature Sets",
        "",
        f"- `all_31`: {feature_counts['all_31']}개 feature. 별점 기반 감성 feature 포함.",
        f"- `leakage_safe`: {feature_counts['leakage_safe']}개 feature. 별점으로 만든 감성 비율 및 긍정/부정 TF-IDF 제외.",
        "",
        "`all_31` 성능은 rating-derived feature 때문에 낙관적으로 측정될 수 있으므로, 최종 연구 결과에는 `leakage_safe`를 함께 보고해야 합니다.",
        "",
    ]

    for feature_set in ["all_31", "leakage_safe"]:
        ranked = sorted(
            [row for row in summaries if row["feature_set"] == feature_set],
            key=lambda row: row["mse_mean"],
        )
        lines.extend(
            [
                f"## Best Results: {feature_set}",
                "",
                "| Rank | Model | Parameters | MSE | RMSE | MAE |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(ranked[:10], start=1):
            lines.append(
                f"| {rank} | {row['model']} | `{row['params']}` | "
                f"{row['mse_mean']:.6f} | {row['rmse_mean']:.6f} | {row['mae_mean']:.6f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare strong tabular regression models with 5-fold CV.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/extended_models"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, all_features = load_nodes(args.nodes)
    feature_sets = {
        "all_31": all_features,
        "leakage_safe": [column for column in all_features if not is_rating_derived_feature(column)],
    }
    y = nodes[TARGET_COLUMN].to_numpy(dtype=float)
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_output: list[dict[str, Any]] = []
    summary_output: list[dict[str, Any]] = []

    for feature_set_name, features in feature_sets.items():
        x = nodes[features].to_numpy(dtype=float)
        print(f"[RUN] feature_set={feature_set_name}, features={len(features)}")

        for model_name, estimator, grid in model_specs(args.seed):
            best_row: dict[str, Any] | None = None
            best_fold_rows: list[dict[str, float]] = []
            for params in ParameterGrid(grid):
                fold_rows = evaluate_configuration(estimator, params, x, y, cv)
                summary = summarize_fold_rows(feature_set_name, model_name, params, fold_rows)
                if best_row is None or summary["mse_mean"] < best_row["mse_mean"]:
                    best_row = summary
                    best_fold_rows = fold_rows

            assert best_row is not None
            summary_output.append(best_row)
            for row in best_fold_rows:
                fold_output.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "params": best_row["params"],
                        **row,
                    }
                )
            print(
                f"  [BEST] {model_name}: MSE={best_row['mse_mean']:.6f}, "
                f"RMSE={best_row['rmse_mean']:.6f}, MAE={best_row['mae_mean']:.6f}"
            )

        stacking = build_stacking_model(args.seed)
        stacking_rows = evaluate_configuration(stacking, {}, x, y, cv)
        stacking_summary = summarize_fold_rows(feature_set_name, "Stacking Ensemble", {}, stacking_rows)
        summary_output.append(stacking_summary)
        for row in stacking_rows:
            fold_output.append(
                {
                    "feature_set": feature_set_name,
                    "model": "Stacking Ensemble",
                    "params": "",
                    **row,
                }
            )

    summary_output.sort(key=lambda row: (row["feature_set"], row["mse_mean"]))
    write_csv(args.out_dir / "extended_cv_summary.csv", summary_output)
    write_csv(args.out_dir / "extended_cv_fold_metrics.csv", fold_output)
    write_summary_markdown(
        args.out_dir / "extended_model_summary.md",
        summary_output,
        {name: len(features) for name, features in feature_sets.items()},
    )

    print(f"[OK] output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
