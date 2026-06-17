from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from run_small_portion_experiment import BASE_FEATURE_COLUMNS, TARGET_COLUMN


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablate structured and KoBERT feature contributions.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/model/lecture_kobert_embeddings.npz"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/transformer_tabular/transformer_ablation.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    nodes["lecture_id"] = nodes["lecture_id"].astype(str)
    archive = np.load(args.embeddings)
    embedding_by_id = dict(zip(archive["lecture_id"].astype(str), archive["embeddings"]))
    nodes = nodes[nodes["lecture_id"].isin(embedding_by_id)].reset_index(drop=True)

    structured = nodes[BASE_FEATURE_COLUMNS].to_numpy(dtype=float)
    embeddings = np.stack([embedding_by_id[value] for value in nodes["lecture_id"]])
    target = nodes[TARGET_COLUMN].to_numpy(dtype=float)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(nodes))

    results: list[dict[str, Any]] = []
    for mode in ["structured_only", "kobert_pca32_only", "structured+kobert_pca32"]:
        for alpha in [1.0, 10.0, 100.0]:
            metrics = []
            for train_idx, valid_idx in folds:
                structured_scaler = StandardScaler()
                train_structured = structured_scaler.fit_transform(structured[train_idx])
                valid_structured = structured_scaler.transform(structured[valid_idx])

                embedding_scaler = StandardScaler()
                train_embedding = embedding_scaler.fit_transform(embeddings[train_idx])
                valid_embedding = embedding_scaler.transform(embeddings[valid_idx])
                pca = PCA(n_components=32, random_state=args.seed)
                train_embedding = pca.fit_transform(train_embedding)
                valid_embedding = pca.transform(valid_embedding)

                if mode == "structured_only":
                    x_train, x_valid = train_structured, valid_structured
                elif mode == "kobert_pca32_only":
                    x_train, x_valid = train_embedding, valid_embedding
                else:
                    x_train = np.c_[train_structured, train_embedding]
                    x_valid = np.c_[valid_structured, valid_embedding]

                model = Ridge(alpha=alpha)
                model.fit(x_train, target[train_idx])
                prediction = np.clip(model.predict(x_valid), 0.0, 1.0)
                mse = float(mean_squared_error(target[valid_idx], prediction))
                metrics.append(
                    (
                        mse,
                        float(np.sqrt(mse)),
                        float(mean_absolute_error(target[valid_idx], prediction)),
                    )
                )

            results.append(
                {
                    "feature_mode": mode,
                    "model": "Ridge",
                    "alpha": alpha,
                    "mse_mean": round(float(np.mean([value[0] for value in metrics])), 8),
                    "rmse_mean": round(float(np.mean([value[1] for value in metrics])), 8),
                    "mae_mean": round(float(np.mean([value[2] for value in metrics])), 8),
                }
            )

    best = []
    for mode in ["structured_only", "kobert_pca32_only", "structured+kobert_pca32"]:
        candidates = [row for row in results if row["feature_mode"] == mode]
        best.append(min(candidates, key=lambda row: row["mse_mean"]))
    write_csv(args.output, best)
    for row in best:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
