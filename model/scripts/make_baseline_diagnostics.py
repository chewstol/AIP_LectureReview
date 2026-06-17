from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_small_portion_experiment import (
    build_split,
    cosine_similarity,
    load_nodes,
    mse,
)


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float) -> np.ndarray:
    train = np.c_[np.ones((len(x_train), 1)), x_train]
    eval_x = np.c_[np.ones((len(x_eval), 1)), x_eval]
    identity = np.eye(train.shape[1])
    identity[0, 0] = 0.0
    weights = np.linalg.pinv(train.T @ train + alpha * identity) @ train.T @ y_train
    return np.clip(eval_x @ weights, 0.0, 1.0)


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    k: int,
    exclude_self: bool = False,
) -> np.ndarray:
    similarities = cosine_similarity(x_eval, x_train)
    preds = []
    for row_idx in range(len(x_eval)):
        sims = similarities[row_idx].copy()
        if exclude_self and row_idx < len(sims):
            sims[row_idx] = -np.inf
        indices = np.argsort(-sims)[: min(k, len(x_train))]
        weights = np.maximum(sims[indices], 0.0)
        if float(weights.sum()) == 0.0:
            preds.append(float(y_train[indices].mean()))
        else:
            preds.append(float(np.sum(y_train[indices, 0] * weights) / weights.sum()))
    return np.array(preds).reshape(-1, 1)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_line_svg(
    rows: list[dict[str, float | int | str]],
    x_key: str,
    title: str,
    x_label: str,
    path: Path,
    log_x: bool = False,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 35, 45, 75
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = np.array([float(row[x_key]) for row in rows], dtype=float)
    x_values = np.log10(xs) if log_x else xs
    train = np.array([float(row["train_mse"]) for row in rows], dtype=float)
    test = np.array([float(row["test_mse"]) for row in rows], dtype=float)
    y_max = float(max(train.max(), test.max()))
    y_min = 0.0

    def point(xv: float, yv: float) -> tuple[float, float]:
        x = left + (xv - x_values.min()) / max(x_values.max() - x_values.min(), 1e-12) * plot_w
        y = top + (1 - (yv - y_min) / max(y_max - y_min, 1e-12)) * plot_h
        return x, y

    train_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [point(x, y) for x, y in zip(x_values, train)])
    test_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [point(x, y) for x, y in zip(x_values, test)])
    best = min(rows, key=lambda row: float(row["test_mse"]))
    best_x = np.log10(float(best[x_key])) if log_x else float(best[x_key])
    best_px, best_py = point(best_x, float(best["test_mse"]))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
  <polyline points="{train_points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  <polyline points="{test_points}" fill="none" stroke="#dc2626" stroke-width="2"/>
  <circle cx="{best_px:.2f}" cy="{best_py:.2f}" r="5" fill="#dc2626"/>
  <text x="{best_px + 10:.2f}" y="{best_py - 10:.2f}" font-family="Arial" font-size="12">best test {x_key}={best[x_key]}</text>
  <text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>
  <text x="24" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24 {height / 2})">MSE</text>
  <rect x="{width - 210}" y="55" width="170" height="58" fill="white" stroke="#e5e7eb"/>
  <line x1="{width - 195}" y1="78" x2="{width - 155}" y2="78" stroke="#2563eb" stroke-width="3"/>
  <text x="{width - 145}" y="83" font-family="Arial" font-size="13">train MSE</text>
  <line x1="{width - 195}" y1="101" x2="{width - 155}" y2="101" stroke="#dc2626" stroke-width="3"/>
  <text x="{width - 145}" y="106" font-family="Arial" font-size="13">test MSE</text>
  <text x="{left - 10}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="12">{y_max:.4f}</text>
  <text x="{left - 10}" y="{top + plot_h + 4}" text-anchor="end" font-family="Arial" font-size="12">0</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Ridge/KNN baseline diagnostic graphs.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/small_portion"))
    parser.add_argument("--sample-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir
    nodes, feature_columns = load_nodes(Path("data/model/lecture_nodes_with_text.csv"))
    split = build_split(nodes, feature_columns, sample_ratio=args.sample_ratio, test_ratio=args.test_ratio, seed=args.seed)

    ridge_rows: list[dict[str, float | int | str]] = []
    for alpha in [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        train_pred = ridge_predict(split.x_train, split.y_train, split.x_train, alpha)
        test_pred = ridge_predict(split.x_train, split.y_train, split.x_test, alpha)
        ridge_rows.append(
            {
                "model": "ridge",
                "alpha": alpha,
                "train_mse": round(mse(split.y_train, train_pred), 8),
                "test_mse": round(mse(split.y_test, test_pred), 8),
            }
        )

    knn_rows: list[dict[str, float | int | str]] = []
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 45]:
        train_pred = knn_predict(split.x_train, split.y_train, split.x_train, k, exclude_self=True)
        test_pred = knn_predict(split.x_train, split.y_train, split.x_test, k)
        knn_rows.append(
            {
                "model": "content_knn",
                "k": k,
                "train_mse": round(mse(split.y_train, train_pred), 8),
                "test_mse": round(mse(split.y_test, test_pred), 8),
            }
        )

    write_csv(out_dir / "ridge_alpha_diagnostics.csv", ridge_rows)
    write_csv(out_dir / "knn_k_diagnostics.csv", knn_rows)
    make_line_svg(
        [row for row in ridge_rows if float(row["alpha"]) > 0],
        "alpha",
        "Ridge Diagnostics: Regularization Strength",
        "alpha (log scale)",
        out_dir / "ridge_alpha_diagnostics.svg",
        log_x=True,
    )
    make_line_svg(
        knn_rows,
        "k",
        "Content KNN Diagnostics: Number of Neighbors",
        "k",
        out_dir / "knn_k_diagnostics.svg",
    )

    print(f"[OK] ridge diagnostics: {(out_dir / 'ridge_alpha_diagnostics.csv').resolve()}")
    print(f"[OK] knn diagnostics: {(out_dir / 'knn_k_diagnostics.csv').resolve()}")
    print(f"[OK] ridge graph: {(out_dir / 'ridge_alpha_diagnostics.svg').resolve()}")
    print(f"[OK] knn graph: {(out_dir / 'knn_k_diagnostics.svg').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
