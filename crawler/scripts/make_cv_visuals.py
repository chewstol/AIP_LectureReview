from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("data/experiments/full_cv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_model_comparison_svg(rows: list[dict[str, str]], path: Path) -> None:
    best_by_model = {}
    for row in rows:
        model = row["model"]
        if model not in best_by_model or float(row["mse_mean"]) < float(best_by_model[model]["mse_mean"]):
            best_by_model[model] = row

    ordered = sorted(best_by_model.values(), key=lambda row: float(row["mse_mean"]))
    width, height = 960, 460
    left, right, top, bottom = 230, 40, 48, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(float(row["mse_mean"]) for row in ordered)
    bar_h = 42
    gap = 28
    colors = ["#2563eb", "#0891b2", "#f97316", "#64748b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="28" text-anchor="middle" font-family="Arial" font-size="18">5-fold CV: Best Hyperparameter per Model</text>',
    ]
    for idx, row in enumerate(ordered):
        y = top + idx * (bar_h + gap)
        value = float(row["mse_mean"])
        bar_w = value / max_value * plot_w
        label = f"{row['model']} ({row['params'] or 'none'})"
        parts.extend(
            [
                f'<text x="{left-12}" y="{y+28}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" fill="{colors[idx % len(colors)]}" rx="4"/>',
                f'<text x="{left+bar_w+8:.1f}" y="{y+28}" font-family="Arial" font-size="13">MSE {value:.4f}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left}" y="{height-28}" font-family="Arial" font-size="12">lower is better</text>',
            "</svg>",
        ]
    )
    write_text(path, "\n".join(parts))


def make_hparam_svg(rows: list[dict[str, str]], model: str, param_prefix: str, path: Path) -> None:
    filtered = [row for row in rows if row["model"] == model]
    width, height = 900, 500
    left, right, top, bottom = 80, 35, 45, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = []
    ys = []
    labels = []
    for row in filtered:
        params = row["params"]
        value = params.split("=", 1)[1]
        xs.append(float(value))
        ys.append(float(row["mse_mean"]))
        labels.append(value)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    labels = [labels[i] for i in order]
    max_y = max(ys)
    min_y = min(ys)

    def point(idx: int, value: float) -> tuple[float, float]:
        x = left + idx / max(len(xs) - 1, 1) * plot_w
        y = top + (1 - (value - min_y) / max(max_y - min_y, 1e-12)) * plot_h
        return x, y

    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [point(i, v) for i, v in enumerate(ys)])
    best_idx = min(range(len(ys)), key=lambda i: ys[i])
    best_x, best_y = point(best_idx, ys[best_idx])
    ticks = "\n".join(
        f'<text x="{point(i, ys[i])[0]:.2f}" y="{height-38}" text-anchor="middle" font-family="Arial" font-size="11">{labels[i]}</text>'
        for i in range(len(labels))
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{model}: CV MSE by {param_prefix}</text>
  <line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#334155"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#334155"/>
  <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  <circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="5" fill="#dc2626"/>
  <text x="{best_x+10:.2f}" y="{best_y-10:.2f}" font-family="Arial" font-size="12">best {param_prefix}={labels[best_idx]}</text>
  {ticks}
  <text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">{param_prefix}</text>
  <text x="24" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24 {height/2})">Mean CV MSE</text>
  <text x="{left-10}" y="{top+4}" text-anchor="end" font-family="Arial" font-size="12">{max_y:.4f}</text>
  <text x="{left-10}" y="{top+plot_h+4}" text-anchor="end" font-family="Arial" font-size="12">{min_y:.4f}</text>
</svg>
"""
    write_text(path, svg)


def main() -> int:
    rows = read_csv(ROOT / "cv_summary.csv")
    make_model_comparison_svg(rows, ROOT / "cv_model_comparison.svg")
    make_hparam_svg(rows, "Ridge Regression", "alpha", ROOT / "cv_ridge_alpha.svg")
    make_hparam_svg(rows, "Content KNN", "k", ROOT / "cv_knn_k.svg")
    print(f"[OK] {ROOT / 'cv_model_comparison.svg'}")
    print(f"[OK] {ROOT / 'cv_ridge_alpha.svg'}")
    print(f"[OK] {ROOT / 'cv_knn_k.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
