from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import pandas as pd


SELECTED_MODELS = [
    "Ridge",
    "Content KNN",
    "SVR-RBF",
    "Random Forest",
    "Gradient Boosting",
    "MLP",
]

COLORS = {
    "Ridge": "#2563eb",
    "Content KNN": "#64748b",
    "SVR-RBF": "#7c3aed",
    "Random Forest": "#ea580c",
    "Gradient Boosting": "#059669",
    "MLP": "#db2777",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the final six-model comparison.")
    parser.add_argument(
        "--all-models",
        type=Path,
        default=Path("data/experiments/kobert_all_models/kobert_all_models_summary.csv"),
    )
    parser.add_argument(
        "--mlp",
        type=Path,
        default=Path("data/experiments/kobert_mlp/kobert_mlp_best.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/experiments/final_six_models"),
    )
    args = parser.parse_args()

    all_models = pd.read_csv(args.all_models)
    mlp = pd.read_csv(args.mlp)
    combined = pd.concat([all_models, mlp], ignore_index=True)
    selected = combined[combined["model"].isin(SELECTED_MODELS)].copy()
    selected = selected.sort_values("rmse_mean").reset_index(drop=True)
    selected.insert(0, "rank", range(1, len(selected) + 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "six_model_comparison.csv", index=False, encoding="utf-8-sig")

    markdown = [
        "# Final Six-model Comparison",
        "",
        "- input: structured 10 + KoBERT PCA32",
        "- validation: 5-fold cross-validation",
        "",
        "| Rank | Model | MSE | RMSE | MAE |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        markdown.append(
            f"| {int(row['rank'])} | {row['model']} | "
            f"{row['mse_mean']:.6f} | {row['rmse_mean']:.6f} | {row['mae_mean']:.6f} |"
        )
    (args.out_dir / "six_model_comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    width, height = 1050, 485
    left, right, top = 270, 105, 92
    chart_width = width - left - right
    maximum = float(selected["rmse_mean"].max()) * 1.08
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="39" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">Final model comparison</text>',
        '<text x="32" y="67" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">Structured 10 + KoBERT PCA32, 5-fold CV, lower RMSE is better</text>',
    ]
    for index, row in selected.iterrows():
        y = top + index * 57
        model = str(row["model"])
        value = float(row["rmse_mean"])
        bar_width = chart_width * value / maximum
        svg.extend(
            [
                f'<text x="{left - 14}" y="{y + 20}" text-anchor="end" font-family="Arial, sans-serif" font-size="16" fill="#111827">{html.escape(model)}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="29" rx="4" fill="#eef2f7"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="29" rx="4" fill="{COLORS[model]}"/>',
                f'<text x="{left + bar_width + 10:.2f}" y="{y + 20}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">{value:.5f}</text>',
            ]
        )
    svg.append("</svg>")
    (args.out_dir / "six_model_comparison_rmse.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(selected[["rank", "model", "mse_mean", "rmse_mean", "mae_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
