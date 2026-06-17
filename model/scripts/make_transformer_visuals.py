from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


COLORS = {
    "structured_only": "#2563eb",
    "kobert_pca32_only": "#7c3aed",
    "structured+kobert_pca32": "#059669",
}


LABELS = {
    "structured_only": "Structured 10",
    "kobert_pca32_only": "KoBERT PCA32",
    "structured+kobert_pca32": "Structured 10 + KoBERT PCA32",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create KoBERT ablation chart.")
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path("data/experiments/transformer_tabular/transformer_ablation.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/transformer_tabular/kobert_ablation_rmse.svg"),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.ablation).sort_values("rmse_mean")
    width, height = 980, 330
    left, chart_width = 320, 520
    maximum = float(data["rmse_mean"].max()) * 1.12
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">KoBERT feature ablation</text>',
        '<text x="32" y="69" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">Ridge regression, 5-fold cross-validation, lower RMSE is better</text>',
    ]
    for index, row in data.reset_index(drop=True).iterrows():
        y = 105 + index * 66
        mode = str(row["feature_mode"])
        value = float(row["rmse_mean"])
        bar_width = chart_width * value / maximum
        svg.extend(
            [
                f'<text x="{left - 14}" y="{y + 22}" text-anchor="end" font-family="Arial, sans-serif" font-size="16" fill="#111827">{html.escape(LABELS[mode])}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="32" rx="4" fill="#eef2f7"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="32" rx="4" fill="{COLORS[mode]}"/>',
                f'<text x="{left + bar_width + 10:.2f}" y="{y + 22}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">{value:.5f}</text>',
            ]
        )
    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"[OK] output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
