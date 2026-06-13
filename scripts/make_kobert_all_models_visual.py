from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


COLORS = {
    "Ridge": "#2563eb",
    "XGBoost": "#059669",
    "Extra Trees": "#d97706",
    "Gradient Boosting": "#16a34a",
    "Histogram Gradient Boosting": "#0d9488",
    "LightGBM": "#65a30d",
    "Kernel Ridge-RBF": "#7c3aed",
    "Random Forest": "#ea580c",
    "SVR-RBF": "#db2777",
    "Content KNN": "#64748b",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create KoBERT all-model comparison chart.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/experiments/kobert_all_models/kobert_all_models_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/kobert_all_models/kobert_all_models_rmse.svg"),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.summary).sort_values("rmse_mean").reset_index(drop=True)
    width = 1080
    height = 120 + len(data) * 54
    left, right = 300, 110
    chart_width = width - left - right
    maximum = float(data["rmse_mean"].max()) * 1.08
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="40" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">KoBERT PCA32 + structured 10</text>',
        '<text x="32" y="68" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">5-fold cross-validation, lower RMSE is better</text>',
    ]
    for index, row in data.iterrows():
        y = 98 + index * 54
        model = str(row["model"])
        value = float(row["rmse_mean"])
        bar_width = chart_width * value / maximum
        svg.extend(
            [
                f'<text x="{left - 14}" y="{y + 19}" text-anchor="end" font-family="Arial, sans-serif" font-size="15" fill="#111827">{html.escape(model)}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="27" rx="4" fill="#eef2f7"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="27" rx="4" fill="{COLORS.get(model, "#475569")}"/>',
                f'<text x="{left + bar_width + 10:.2f}" y="{y + 19}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">{value:.5f}</text>',
            ]
        )
    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"[OK] output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
