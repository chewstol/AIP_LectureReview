from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from run_extended_model_experiment import is_rating_derived_feature
from run_small_portion_experiment import TARGET_COLUMN, load_nodes, mae, mse, rmse


ALPHAS = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
GAMMAS = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def color(value: float, low: float, high: float) -> str:
    ratio = 0.0 if high <= low else (value - low) / (high - low)
    ratio = min(max(ratio, 0.0), 1.0)
    start = np.array([220, 252, 231], dtype=float)
    end = np.array([254, 202, 202], dtype=float)
    rgb = np.round(start + ratio * (end - start)).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def text_color(background_value: float, low: float, high: float) -> str:
    ratio = 0.0 if high <= low else (background_value - low) / (high - low)
    return "#111827" if ratio < 0.65 else "#7f1d1d"


def make_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = {(row["alpha"], row["gamma"]): row for row in rows}
    best = min(rows, key=lambda row: row["valid_rmse_mean"])
    values = [float(row["valid_rmse_mean"]) for row in rows]
    low, high = min(values), max(values)

    width, height = 1180, 760
    left, top = 155, 112
    cell_width, cell_height = 94, 57
    grid_width = len(GAMMAS) * cell_width
    grid_height = len(ALPHAS) * cell_height
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="34" y="42" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#111827">Kernel Ridge hyperparameter sweet spot</text>',
        '<text x="34" y="72" font-family="Arial, sans-serif" font-size="15" fill="#4b5563">Structured 10 + category TF-IDF 6 · 5-fold CV · cell value = validation RMSE</text>',
        f'<text x="{left + grid_width / 2}" y="{height - 42}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#374151">RBF gamma</text>',
        f'<text x="34" y="{top + grid_height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#374151" transform="rotate(-90 34 {top + grid_height / 2})">Regularization alpha</text>',
    ]

    for column, gamma in enumerate(GAMMAS):
        x = left + column * cell_width
        svg.append(
            f'<text x="{x + cell_width / 2}" y="{top - 14}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{gamma:g}</text>'
        )

    for row_index, alpha in enumerate(ALPHAS):
        y = top + row_index * cell_height
        svg.append(
            f'<text x="{left - 15}" y="{y + cell_height / 2 + 5}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#374151">{alpha:g}</text>'
        )
        for column, gamma in enumerate(GAMMAS):
            x = left + column * cell_width
            record = lookup[(alpha, gamma)]
            value = float(record["valid_rmse_mean"])
            is_best = alpha == best["alpha"] and gamma == best["gamma"]
            svg.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{color(value, low, high)}" stroke="{"#059669" if is_best else "#ffffff"}" stroke-width="{"4" if is_best else "2"}"/>',
                    f'<text x="{x + cell_width / 2}" y="{y + cell_height / 2 + 5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="{"700" if is_best else "500"}" fill="{text_color(value, low, high)}">{value:.4f}</text>',
                ]
            )

    box_x = left + grid_width + 35
    box_y = top + 55
    svg.extend(
        [
            f'<rect x="{box_x}" y="{box_y}" width="225" height="126" rx="6" fill="#ecfdf5" stroke="#059669" stroke-width="2"/>',
            f'<text x="{box_x + 15}" y="{box_y + 29}" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#065f46">Best combination</text>',
            f'<text x="{box_x + 15}" y="{box_y + 58}" font-family="Arial, sans-serif" font-size="15" fill="#065f46">alpha = {best["alpha"]:g}</text>',
            f'<text x="{box_x + 15}" y="{box_y + 82}" font-family="Arial, sans-serif" font-size="15" fill="#065f46">gamma = {best["gamma"]:g}</text>',
            f'<text x="{box_x + 15}" y="{box_y + 106}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#065f46">RMSE = {best["valid_rmse_mean"]:.5f}</text>',
            f'<text x="{box_x}" y="{box_y + 175}" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Green: lower error</text>',
            f'<text x="{box_x}" y="{box_y + 198}" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Red: higher error</text>',
        ]
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def make_gamma_slice(rows: list[dict[str, Any]], path: Path) -> None:
    best = min(rows, key=lambda row: row["valid_rmse_mean"])
    selected = sorted(
        [row for row in rows if row["alpha"] == best["alpha"]],
        key=lambda row: row["gamma"],
    )
    width, height = 1000, 560
    left, right, top, bottom = 85, 50, 90, 85
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = [row["valid_rmse_mean"] for row in selected]
    low = min(values) * 0.98
    high = max(values) * 1.02
    log_low = math.log10(min(GAMMAS))
    log_high = math.log10(max(GAMMAS))

    def px(gamma: float) -> float:
        return left + (math.log10(gamma) - log_low) / (log_high - log_low) * chart_width

    def py(value: float) -> float:
        return top + (high - value) / (high - low) * chart_height

    points = " ".join(f"{px(row['gamma']):.2f},{py(row['valid_rmse_mean']):.2f}" for row in selected)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="40" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">Gamma sensitivity at alpha={best["alpha"]:g}</text>',
        '<text x="32" y="68" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">5-fold validation RMSE · gamma shown on log scale</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#f8fafc" stroke="#d1d5db"/>',
        f'<polyline points="{points}" fill="none" stroke="#7c3aed" stroke-width="3"/>',
    ]
    for row in selected:
        x = px(row["gamma"])
        y = py(row["valid_rmse_mean"])
        is_best = row["gamma"] == best["gamma"]
        svg.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{"8" if is_best else "5"}" fill="{"#059669" if is_best else "#7c3aed"}"/>',
                f'<text x="{x:.2f}" y="{top + chart_height + 27}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{row["gamma"]:g}</text>',
            ]
        )
    svg.extend(
        [
            f'<text x="{left + chart_width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#374151">RBF gamma</text>',
            f'<text x="26" y="{top + chart_height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#374151" transform="rotate(-90 26 {top + chart_height / 2})">Validation RMSE</text>',
        ]
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find Kernel Ridge alpha/gamma sweet spot for TF-IDF 16 features.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/tfidf16_kernel_ridge_tuning"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodes, all_features = load_nodes(args.nodes)
    features = [column for column in all_features if not is_rating_derived_feature(column)]
    x = nodes[features].to_numpy(dtype=float)
    y = nodes[TARGET_COLUMN].to_numpy(dtype=float)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(nodes))

    prepared = []
    for train_idx, valid_idx in folds:
        scaler = StandardScaler()
        prepared.append(
            (
                scaler.fit_transform(x[train_idx]),
                scaler.transform(x[valid_idx]),
                train_idx,
                valid_idx,
            )
        )

    rows = []
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            train_values = []
            valid_values = []
            valid_mse_values = []
            valid_mae_values = []
            for x_train, x_valid, train_idx, valid_idx in prepared:
                model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
                model.fit(x_train, y[train_idx])
                train_values.append(rmse(y[train_idx], np.clip(model.predict(x_train), 0.0, 1.0)))
                valid_prediction = np.clip(model.predict(x_valid), 0.0, 1.0)
                valid_values.append(rmse(y[valid_idx], valid_prediction))
                valid_mse_values.append(mse(y[valid_idx], valid_prediction))
                valid_mae_values.append(mae(y[valid_idx], valid_prediction))
            rows.append(
                {
                    "alpha": alpha,
                    "gamma": gamma,
                    "train_rmse_mean": round(float(np.mean(train_values)), 8),
                    "train_rmse_std": round(float(np.std(train_values)), 8),
                    "valid_rmse_mean": round(float(np.mean(valid_values)), 8),
                    "valid_rmse_std": round(float(np.std(valid_values)), 8),
                    "valid_mse_mean": round(float(np.mean(valid_mse_values)), 8),
                    "valid_mae_mean": round(float(np.mean(valid_mae_values)), 8),
                    "generalization_gap": round(float(np.mean(valid_values) - np.mean(train_values)), 8),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "kernel_ridge_grid_search.csv", rows)
    make_heatmap(rows, args.out_dir / "kernel_ridge_alpha_gamma_heatmap.svg")
    make_gamma_slice(rows, args.out_dir / "kernel_ridge_gamma_sensitivity.svg")
    best = min(rows, key=lambda row: row["valid_rmse_mean"])
    print(f"[BEST] alpha={best['alpha']:g}, gamma={best['gamma']:g}, RMSE={best['valid_rmse_mean']:.8f}")
    print(f"[OK] output={args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
