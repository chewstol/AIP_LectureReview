from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


COLORS = {
    "Ridge": "#2563eb",
    "Kernel Ridge-RBF": "#7c3aed",
    "Histogram Gradient Boosting": "#059669",
    "Random Forest": "#ea580c",
    "Extra Trees": "#d97706",
    "Gradient Boosting": "#16a34a",
    "SVR-RBF": "#db2777",
    "Stacking Ensemble": "#64748b",
}


def make_chart(data: pd.DataFrame, feature_set: str, output: Path) -> None:
    rows = data[data["feature_set"] == feature_set].sort_values("rmse_mean").reset_index(drop=True)
    width = 1040
    height = 100 + len(rows) * 58
    left = 270
    right = 110
    chart_width = width - left - right
    maximum = float(rows["rmse_mean"].max()) * 1.12

    title = "31 features (rating-derived features included)" if feature_set == "all_31" else "16 leakage-safe features"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        '<text x="32" y="65" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">5-fold cross-validation, lower RMSE is better</text>',
    ]

    for index, row in rows.iterrows():
        y = 92 + index * 58
        model = str(row["model"])
        value = float(row["rmse_mean"])
        bar_width = chart_width * value / maximum
        color = COLORS.get(model, "#475569")
        svg.extend(
            [
                f'<text x="{left - 14}" y="{y + 20}" text-anchor="end" font-family="Arial, sans-serif" font-size="15" fill="#111827">{html.escape(model)}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="28" rx="4" fill="#eef2f7"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="28" rx="4" fill="{color}"/>',
                f'<text x="{left + bar_width + 10:.2f}" y="{y + 20}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">{value:.5f}</text>',
            ]
        )

    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def make_legacy_comparison(extended: pd.DataFrame, legacy_path: Path, output: Path) -> None:
    safe = extended[extended["feature_set"] == "leakage_safe"].copy()
    legacy = pd.read_csv(legacy_path)
    combined = pd.concat([safe, legacy], ignore_index=True)
    combined = combined.sort_values("rmse_mean").drop_duplicates("model", keep="first")
    combined.to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    width = 1040
    height = 120 + len(combined) * 58
    left = 270
    right = 110
    chart_width = width - left - right
    maximum = float(combined["rmse_mean"].max()) * 1.10
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">Leakage-safe model comparison</text>',
        '<text x="32" y="65" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">16 features, 5-fold cross-validation, lower RMSE is better</text>',
    ]
    for index, row in combined.reset_index(drop=True).iterrows():
        y = 92 + index * 58
        model = str(row["model"])
        value = float(row["rmse_mean"])
        bar_width = chart_width * value / maximum
        color = COLORS.get(model, "#475569")
        svg.extend(
            [
                f'<text x="{left - 14}" y="{y + 20}" text-anchor="end" font-family="Arial, sans-serif" font-size="15" fill="#111827">{html.escape(model)}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="28" rx="4" fill="#eef2f7"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="28" rx="4" fill="{color}"/>',
                f'<text x="{left + bar_width + 10:.2f}" y="{y + 20}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">{value:.5f}</text>',
            ]
        )
    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create extended model comparison SVGs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/experiments/extended_models/extended_cv_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/experiments/extended_models"),
    )
    parser.add_argument(
        "--legacy-best",
        type=Path,
        default=Path("data/experiments/leakage_safe_legacy/legacy_best_models.csv"),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_chart(data, "all_31", args.out_dir / "extended_models_all31_rmse.svg")
    make_chart(data, "leakage_safe", args.out_dir / "extended_models_leakage_safe_rmse.svg")
    if args.legacy_best.exists():
        make_legacy_comparison(
            data,
            args.legacy_best,
            args.out_dir / "leakage_safe_all_models_rmse.svg",
        )
    print(f"[OK] output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
