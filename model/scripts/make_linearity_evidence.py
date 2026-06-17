from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "grading_generous_score": "Grading generous",
    "grading_strict_score": "Grading strict",
    "assignment_low_score": "Low assignment load",
    "assignment_high_score": "High assignment load",
    "teamwork_low_score": "Low teamwork load",
    "teamwork_high_score": "High teamwork load",
    "attendance_light_score": "Light attendance",
    "attendance_strict_score": "Strict attendance",
    "exam_light_score": "Light exam load",
    "exam_heavy_score": "Heavy exam load",
}

FEATURES = list(FEATURE_LABELS)
TARGET = "rating_average_norm"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation_rows(nodes: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for feature in FEATURES:
        x = nodes[feature].to_numpy(dtype=float)
        y = nodes[TARGET].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        prediction = slope * x + intercept
        total = float(np.sum((y - y.mean()) ** 2))
        residual = float(np.sum((y - prediction) ** 2))
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 6),
                "univariate_linear_r2": round(1.0 - residual / total, 6),
                "slope": round(float(slope), 6),
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["pearson_r"])), reverse=True)


def scale(value: float, low: float, high: float, start: float, length: float) -> float:
    if high <= low:
        return start + length / 2
    return start + (value - low) / (high - low) * length


def make_correlation_chart(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 1040, 690
    left, right, top = 260, 90, 105
    chart_width = width - left - right
    row_height = 50
    center = left + chart_width / 2
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="42" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#111827">Structured feature correlation with average rating</text>',
        '<text x="32" y="72" font-family="Arial, sans-serif" font-size="15" fill="#4b5563">Pearson correlation, 753 lectures</text>',
        f'<line x1="{center}" y1="{top - 18}" x2="{center}" y2="{height - 55}" stroke="#9ca3af" stroke-width="1"/>',
        f'<text x="{left}" y="{height - 23}" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">-1.0</text>',
        f'<text x="{center}" y="{height - 23}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">0</text>',
        f'<text x="{left + chart_width}" y="{height - 23}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">+1.0</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = float(row["pearson_r"])
        end = center + value * chart_width / 2
        x = min(center, end)
        bar_width = abs(end - center)
        color = "#2563eb" if value >= 0 else "#dc2626"
        svg.extend(
            [
                f'<text x="{left - 15}" y="{y + 20}" text-anchor="end" font-family="Arial, sans-serif" font-size="15" fill="#111827">{html.escape(str(row["label"]))}</text>',
                f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="27" rx="3" fill="{color}"/>',
                f'<text x="{end + (8 if value >= 0 else -8):.2f}" y="{y + 19}" text-anchor="{"start" if value >= 0 else "end"}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">{value:+.3f}</text>',
            ]
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def scatter_panel(
    nodes: pd.DataFrame,
    feature: str,
    panel_x: float,
    panel_y: float,
    panel_width: float,
    panel_height: float,
) -> list[str]:
    x = nodes[feature].to_numpy(dtype=float)
    y = nodes[TARGET].to_numpy(dtype=float) * 5.0
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    plot_left = panel_x + 56
    plot_top = panel_y + 48
    plot_width = panel_width - 78
    plot_height = panel_height - 92
    x_low, x_high = 0.0, 1.0
    y_low, y_high = max(1.0, float(y.min()) - 0.1), 5.0
    output = [
        f'<text x="{panel_x + 10}" y="{panel_y + 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#111827">{html.escape(FEATURE_LABELS[feature])}</text>',
        f'<text x="{panel_x + panel_width - 10}" y="{panel_y + 24}" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">r = {correlation:+.3f}</text>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#d1d5db"/>',
    ]
    for point_x, point_y in zip(x, y):
        cx = scale(float(point_x), x_low, x_high, plot_left, plot_width)
        cy = plot_top + plot_height - scale(float(point_y), y_low, y_high, 0, plot_height)
        output.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.2" fill="#64748b" fill-opacity="0.32"/>')

    line_y1 = slope * x_low + intercept
    line_y2 = slope * x_high + intercept
    sx1 = plot_left
    sx2 = plot_left + plot_width
    sy1 = plot_top + plot_height - scale(float(line_y1), y_low, y_high, 0, plot_height)
    sy2 = plot_top + plot_height - scale(float(line_y2), y_low, y_high, 0, plot_height)
    output.extend(
        [
            f'<line x1="{sx1:.2f}" y1="{sy1:.2f}" x2="{sx2:.2f}" y2="{sy2:.2f}" stroke="#2563eb" stroke-width="3"/>',
            f'<text x="{plot_left + plot_width / 2}" y="{panel_y + panel_height - 8}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">Feature score (0-1)</text>',
            f'<text x="{panel_x + 15}" y="{plot_top + plot_height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#4b5563" transform="rotate(-90 {panel_x + 15} {plot_top + plot_height / 2})">Average rating (1-5)</text>',
        ]
    )
    return output


def make_scatter_chart(nodes: pd.DataFrame, path: Path) -> None:
    width, height = 1180, 800
    margin, gap = 35, 22
    panel_width = (width - margin * 2 - gap) / 2
    panel_height = 330
    selected = [
        "grading_generous_score",
        "grading_strict_score",
        "assignment_low_score",
        "assignment_high_score",
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="35" y="40" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#111827">Linear trends in structured lecture features</text>',
        '<text x="35" y="69" font-family="Arial, sans-serif" font-size="15" fill="#4b5563">Each point is one lecture; blue line is ordinary least-squares trend</text>',
    ]
    for index, feature in enumerate(selected):
        row, column = divmod(index, 2)
        panel_x = margin + column * (panel_width + gap)
        panel_y = 90 + row * (panel_height + gap)
        svg.extend(scatter_panel(nodes, feature, panel_x, panel_y, panel_width, panel_height))
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create evidence plots for Ridge baseline selection.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/linearity"))
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes).dropna(subset=[TARGET, *FEATURES])
    rows = correlation_rows(nodes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "structured_feature_correlations.csv", rows)
    make_correlation_chart(rows, args.out_dir / "structured_feature_correlations.svg")
    make_scatter_chart(nodes, args.out_dir / "structured_feature_scatter_trends.svg")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[OK] output={args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
