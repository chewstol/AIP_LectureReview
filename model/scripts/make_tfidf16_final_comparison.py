from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


MODELS = [
    "Kernel Ridge-RBF",
    "Ridge",
    "Content KNN",
    "SVR-RBF",
    "Random Forest",
    "Gradient Boosting",
    "MLP",
]

COLORS = {
    "Kernel Ridge-RBF": "#059669",
    "Ridge": "#2563eb",
    "Content KNN": "#64748b",
    "SVR-RBF": "#7c3aed",
    "Random Forest": "#ea580c",
    "Gradient Boosting": "#16a34a",
    "MLP": "#db2777",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create final TF-IDF 16-feature model comparison.")
    parser.add_argument(
        "--extended",
        type=Path,
        default=Path("data/experiments/extended_models/extended_cv_summary.csv"),
    )
    parser.add_argument(
        "--knn",
        type=Path,
        default=Path("data/experiments/leakage_safe_legacy/legacy_best_models.csv"),
    )
    parser.add_argument(
        "--mlp",
        type=Path,
        default=Path("data/experiments/tfidf16_mlp/tfidf16_mlp_best.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/experiments/tfidf16_final"),
    )
    args = parser.parse_args()

    extended = pd.read_csv(args.extended)
    extended = extended[extended["feature_set"] == "leakage_safe"]
    knn = pd.read_csv(args.knn)
    knn = knn[knn["model"] == "Content KNN"]
    mlp = pd.read_csv(args.mlp)
    combined = pd.concat([extended, knn, mlp], ignore_index=True)
    combined = combined[combined["model"].isin(MODELS)]
    combined = combined.sort_values("mse_mean").drop_duplicates("model", keep="first")
    combined = combined.sort_values("rmse_mean").reset_index(drop=True)
    combined.insert(0, "rank", range(1, len(combined) + 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_dir / "tfidf16_model_comparison.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# TF-IDF 16-feature Model Comparison",
        "",
        "- input: structured 10 + category TF-IDF 6",
        "- validation: 5-fold cross-validation",
        "- final model: Kernel Ridge-RBF",
        "",
        "| Rank | Model | Best parameters | MSE | RMSE | MAE |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for _, row in combined.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['model']} | `{row['params']}` | "
            f"{row['mse_mean']:.6f} | {row['rmse_mean']:.6f} | {row['mae_mean']:.6f} |"
        )
    (args.out_dir / "tfidf16_model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    width, height = 1080, 550
    left, right, top = 285, 110, 100
    chart_width = width - left - right
    maximum = float(combined["rmse_mean"].max()) * 1.08
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="40" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#111827">TF-IDF feature model comparison</text>',
        '<text x="32" y="69" font-family="Arial, sans-serif" font-size="15" fill="#4b5563">Structured 10 + category TF-IDF 6 · 5-fold CV · lower RMSE is better</text>',
    ]
    for index, row in combined.iterrows():
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
    (args.out_dir / "tfidf16_model_comparison_rmse.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(combined[["rank", "model", "mse_mean", "rmse_mean", "mae_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
