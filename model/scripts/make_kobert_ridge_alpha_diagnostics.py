from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold

from run_kobert_all_models import load_data, prepare_folds


ALPHAS = [0.0, 0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def x_position(alpha: float, left: float, width: float) -> float:
    # Keep alpha=0 on the chart while using a logarithmic visual scale.
    transformed = math.log10(alpha + 0.001)
    low = math.log10(0.001)
    high = math.log10(1000.001)
    return left + (transformed - low) / (high - low) * width


def y_position(value: float, low: float, high: float, top: float, height: float) -> float:
    return top + (high - value) / (high - low) * height


def make_svg(rows: list[dict[str, float]], path: Path) -> None:
    width, height = 1120, 650
    left, right, top, bottom = 95, 55, 95, 105
    chart_width = width - left - right
    chart_height = height - top - bottom

    train_values = [row["train_rmse_mean"] for row in rows]
    valid_values = [row["valid_rmse_mean"] for row in rows]
    valid_upper = [row["valid_rmse_mean"] + row["valid_rmse_std"] for row in rows]
    y_low = min(train_values + valid_values) * 0.90
    y_high = max(valid_upper) * 1.08
    best = min(rows, key=lambda row: row["valid_rmse_mean"])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="34" y="40" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#111827">Ridge regularization sweet spot</text>',
        '<text x="34" y="69" font-family="Arial, sans-serif" font-size="15" fill="#4b5563">Structured 10 + KoBERT PCA32 · 5-fold cross-validation</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#f8fafc" stroke="#d1d5db"/>',
    ]

    for tick in np.linspace(y_low, y_high, 6):
        y = y_position(float(tick), y_low, y_high, top, chart_height)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{tick:.3f}</text>',
            ]
        )

    label_alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for alpha in label_alphas:
        x = x_position(alpha, left, chart_width)
        label = "0" if alpha == 0 else f"{alpha:g}"
        svg.extend(
            [
                f'<line x1="{x:.2f}" y1="{top + chart_height}" x2="{x:.2f}" y2="{top + chart_height + 6}" stroke="#6b7280"/>',
                f'<text x="{x:.2f}" y="{top + chart_height + 27}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{label}</text>',
            ]
        )

    train_points = []
    valid_points = []
    for row in rows:
        x = x_position(row["alpha"], left, chart_width)
        train_y = y_position(row["train_rmse_mean"], y_low, y_high, top, chart_height)
        valid_y = y_position(row["valid_rmse_mean"], y_low, y_high, top, chart_height)
        train_points.append(f"{x:.2f},{train_y:.2f}")
        valid_points.append(f"{x:.2f},{valid_y:.2f}")

        error_top = y_position(
            row["valid_rmse_mean"] + row["valid_rmse_std"],
            y_low,
            y_high,
            top,
            chart_height,
        )
        error_bottom = y_position(
            row["valid_rmse_mean"] - row["valid_rmse_std"],
            y_low,
            y_high,
            top,
            chart_height,
        )
        svg.extend(
            [
                f'<line x1="{x:.2f}" y1="{error_top:.2f}" x2="{x:.2f}" y2="{error_bottom:.2f}" stroke="#dc2626" stroke-opacity="0.45"/>',
                f'<line x1="{x - 4:.2f}" y1="{error_top:.2f}" x2="{x + 4:.2f}" y2="{error_top:.2f}" stroke="#dc2626" stroke-opacity="0.45"/>',
                f'<line x1="{x - 4:.2f}" y1="{error_bottom:.2f}" x2="{x + 4:.2f}" y2="{error_bottom:.2f}" stroke="#dc2626" stroke-opacity="0.45"/>',
            ]
        )

    svg.extend(
        [
            f'<polyline points="{" ".join(train_points)}" fill="none" stroke="#2563eb" stroke-width="3"/>',
            f'<polyline points="{" ".join(valid_points)}" fill="none" stroke="#dc2626" stroke-width="3"/>',
        ]
    )

    for row in rows:
        x = x_position(row["alpha"], left, chart_width)
        train_y = y_position(row["train_rmse_mean"], y_low, y_high, top, chart_height)
        valid_y = y_position(row["valid_rmse_mean"], y_low, y_high, top, chart_height)
        svg.extend(
            [
                f'<circle cx="{x:.2f}" cy="{train_y:.2f}" r="4" fill="#2563eb"/>',
                f'<circle cx="{x:.2f}" cy="{valid_y:.2f}" r="4" fill="#dc2626"/>',
            ]
        )

    best_x = x_position(best["alpha"], left, chart_width)
    best_y = y_position(best["valid_rmse_mean"], y_low, y_high, top, chart_height)
    box_x = min(best_x + 24, width - 300)
    box_y = max(top + 18, best_y - 90)
    svg.extend(
        [
            f'<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="9" fill="none" stroke="#059669" stroke-width="4"/>',
            f'<line x1="{best_x + 8:.2f}" y1="{best_y - 7:.2f}" x2="{box_x:.2f}" y2="{box_y + 34:.2f}" stroke="#059669" stroke-width="2"/>',
            f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="250" height="70" rx="6" fill="#ecfdf5" stroke="#059669"/>',
            f'<text x="{box_x + 14:.2f}" y="{box_y + 27:.2f}" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#065f46">Sweet spot: alpha = {best["alpha"]:g}</text>',
            f'<text x="{box_x + 14:.2f}" y="{box_y + 52:.2f}" font-family="Arial, sans-serif" font-size="14" fill="#065f46">CV RMSE = {best["valid_rmse_mean"]:.5f}</text>',
            f'<text x="{left + chart_width / 2}" y="{height - 43}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#374151">Ridge alpha (log scale)</text>',
            f'<text x="27" y="{top + chart_height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#374151" transform="rotate(-90 27 {top + chart_height / 2})">RMSE</text>',
            f'<line x1="{width - 295}" y1="40" x2="{width - 255}" y2="40" stroke="#2563eb" stroke-width="3"/>',
            f'<text x="{width - 245}" y="45" font-family="Arial, sans-serif" font-size="14" fill="#111827">Train</text>',
            f'<line x1="{width - 175}" y1="40" x2="{width - 135}" y2="40" stroke="#dc2626" stroke-width="3"/>',
            f'<text x="{width - 125}" y="45" font-family="Arial, sans-serif" font-size="14" fill="#111827">Validation</text>',
        ]
    )

    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Ridge alpha sweet spot for final KoBERT features.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/model/lecture_kobert_embeddings.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/ridge_alpha_final"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, structured, embeddings, target = load_data(args.nodes, args.embeddings)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(nodes))
    folds = prepare_folds(structured, embeddings, splits, pca_components=32)

    rows = []
    for alpha in ALPHAS:
        train_rmse = []
        valid_rmse = []
        coefficient_norms = []
        for x_train, x_valid, train_idx, valid_idx in folds:
            model = Ridge(alpha=alpha)
            model.fit(x_train, target[train_idx])
            train_prediction = np.clip(model.predict(x_train), 0.0, 1.0)
            valid_prediction = np.clip(model.predict(x_valid), 0.0, 1.0)
            train_rmse.append(float(np.sqrt(mean_squared_error(target[train_idx], train_prediction))))
            valid_rmse.append(float(np.sqrt(mean_squared_error(target[valid_idx], valid_prediction))))
            coefficient_norms.append(float(np.linalg.norm(model.coef_)))

        rows.append(
            {
                "alpha": alpha,
                "train_rmse_mean": round(float(np.mean(train_rmse)), 8),
                "train_rmse_std": round(float(np.std(train_rmse)), 8),
                "valid_rmse_mean": round(float(np.mean(valid_rmse)), 8),
                "valid_rmse_std": round(float(np.std(valid_rmse)), 8),
                "coefficient_l2_norm_mean": round(float(np.mean(coefficient_norms)), 8),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "ridge_alpha_final_diagnostics.csv", rows)
    make_svg(rows, args.out_dir / "ridge_alpha_sweet_spot.svg")
    best = min(rows, key=lambda row: row["valid_rmse_mean"])
    print(f"[BEST] alpha={best['alpha']:g}, validation RMSE={best['valid_rmse_mean']:.8f}")
    print(f"[OK] output={args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
