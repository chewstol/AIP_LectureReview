from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_metric_bar_svg(metrics: list[dict[str, str]], path: Path) -> None:
    width, height = 960, 520
    left, right, top, bottom = 220, 40, 50, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(row["test_mse"]) for row in metrics]
    max_value = max(values) if values else 1.0
    bar_h = plot_h / max(len(metrics), 1) * 0.62
    gap = plot_h / max(len(metrics), 1) * 0.38

    colors = ["#64748b", "#2563eb", "#0891b2", "#f97316", "#dc2626"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="28" text-anchor="middle" font-family="Arial" font-size="18">Baseline vs Proposed Model - Test MSE</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>',
    ]

    for idx, row in enumerate(metrics):
        y = top + idx * (bar_h + gap)
        value = float(row["test_mse"])
        bar_w = value / max_value * plot_w
        label = row["model"].replace("Baseline: ", "").replace("Proposed: ", "")
        color = colors[idx % len(colors)]
        parts.extend(
            [
                f'<text x="{left-12}" y="{y+bar_h*0.65:.1f}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>',
                f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>',
                f'<text x="{left+bar_w+8:.1f}" y="{y+bar_h*0.65:.1f}" font-family="Arial" font-size="13">{value:.4f}</text>',
            ]
        )

    parts.extend(
        [
            f'<text x="{left}" y="{height-35}" font-family="Arial" font-size="12">lower is better</text>',
            "</svg>",
        ]
    )
    write_text(path, "\n".join(parts))


def make_overfitting_detail_svg(history: list[dict[str, str]], path: Path) -> None:
    best = min(history, key=lambda row: float(row["test_loss"]))
    final = history[-1]
    width, height = 900, 360

    train_drop = float(history[0]["train_loss"]) - float(final["train_loss"])
    test_increase = float(final["test_loss"]) - float(best["test_loss"])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="450" y="32" text-anchor="middle" font-family="Arial" font-size="20">Overfitting Observation</text>
  <rect x="55" y="70" width="230" height="190" fill="#eff6ff" stroke="#bfdbfe" rx="8"/>
  <text x="170" y="105" text-anchor="middle" font-family="Arial" font-size="15" font-weight="bold">Best Epoch</text>
  <text x="170" y="145" text-anchor="middle" font-family="Arial" font-size="34">{best["epoch"]}</text>
  <text x="170" y="185" text-anchor="middle" font-family="Arial" font-size="14">train loss {float(best["train_loss"]):.6f}</text>
  <text x="170" y="215" text-anchor="middle" font-family="Arial" font-size="14">test loss {float(best["test_loss"]):.6f}</text>
  <rect x="335" y="70" width="230" height="190" fill="#fff7ed" stroke="#fed7aa" rx="8"/>
  <text x="450" y="105" text-anchor="middle" font-family="Arial" font-size="15" font-weight="bold">Final Epoch</text>
  <text x="450" y="145" text-anchor="middle" font-family="Arial" font-size="34">{final["epoch"]}</text>
  <text x="450" y="185" text-anchor="middle" font-family="Arial" font-size="14">train loss {float(final["train_loss"]):.6f}</text>
  <text x="450" y="215" text-anchor="middle" font-family="Arial" font-size="14">test loss {float(final["test_loss"]):.6f}</text>
  <rect x="615" y="70" width="230" height="190" fill="#fef2f2" stroke="#fecaca" rx="8"/>
  <text x="730" y="105" text-anchor="middle" font-family="Arial" font-size="15" font-weight="bold">Interpretation</text>
  <text x="730" y="145" text-anchor="middle" font-family="Arial" font-size="14">train loss decreased</text>
  <text x="730" y="172" text-anchor="middle" font-family="Arial" font-size="18">{train_drop:.4f}</text>
  <text x="730" y="207" text-anchor="middle" font-family="Arial" font-size="14">test loss increased after best</text>
  <text x="730" y="234" text-anchor="middle" font-family="Arial" font-size="18">{test_increase:.4f}</text>
  <text x="450" y="315" text-anchor="middle" font-family="Arial" font-size="14">As epochs increase, the model memorizes train data while test performance worsens.</text>
</svg>
"""
    write_text(path, svg)


def make_loss_curve_svg(
    history: list[dict[str, str]],
    path: Path,
    title: str,
    start_epoch: int = 1,
    log_scale: bool = False,
) -> None:
    filtered = [row for row in history if int(row["epoch"]) >= start_epoch]
    width, height = 900, 520
    left, right, top, bottom = 75, 30, 45, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    epochs = [int(row["epoch"]) for row in filtered]
    train = [float(row["train_loss"]) for row in filtered]
    test = [float(row["test_loss"]) for row in filtered]

    if log_scale:
        min_loss = min([value for value in train + test if value > 0])
        max_loss = max(train + test)

        def scale_loss(value: float) -> float:
            low = math.log10(min_loss)
            high = math.log10(max_loss)
            return (math.log10(max(value, min_loss)) - low) / max(high - low, 1e-12)

        y_label = "MSE Loss (log scale)"
    else:
        min_loss = 0.0
        max_loss = max(train + test)

        def scale_loss(value: float) -> float:
            return (value - min_loss) / max(max_loss - min_loss, 1e-12)

        y_label = "MSE Loss"

    def point(epoch: int, loss: float) -> tuple[float, float]:
        x = left + (epoch - epochs[0]) / max(epochs[-1] - epochs[0], 1) * plot_w
        y = top + (1.0 - scale_loss(loss)) * plot_h
        return x, y

    train_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [point(e, l) for e, l in zip(epochs, train)])
    test_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [point(e, l) for e, l in zip(epochs, test)])

    best = min(filtered, key=lambda row: float(row["test_loss"]))
    best_x, best_y = point(int(best["epoch"]), float(best["test_loss"]))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="26" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
  <polyline points="{train_points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  <polyline points="{test_points}" fill="none" stroke="#dc2626" stroke-width="2"/>
  <circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="5" fill="#dc2626"/>
  <text x="{best_x + 10:.2f}" y="{best_y - 10:.2f}" font-family="Arial" font-size="12">best test epoch {best["epoch"]}</text>
  <text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">Epoch</text>
  <text x="24" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24 {height / 2})">{y_label}</text>
  <rect x="{width - 215}" y="54" width="175" height="58" fill="white" stroke="#e5e7eb"/>
  <line x1="{width - 200}" y1="77" x2="{width - 160}" y2="77" stroke="#2563eb" stroke-width="3"/>
  <text x="{width - 150}" y="82" font-family="Arial" font-size="13">train loss</text>
  <line x1="{width - 200}" y1="100" x2="{width - 160}" y2="100" stroke="#dc2626" stroke-width="3"/>
  <text x="{width - 150}" y="105" font-family="Arial" font-size="13">test loss</text>
  <text x="{left}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{epochs[0]}</text>
  <text x="{left + plot_w}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{epochs[-1]}</text>
  <text x="{left - 10}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="12">{max_loss:.4g}</text>
  <text x="{left - 10}" y="{top + plot_h + 4}" text-anchor="end" font-family="Arial" font-size="12">{min_loss:.4g}</text>
</svg>
"""
    write_text(path, svg)


def make_loss_points_svg(history: list[dict[str, str]], path: Path, start_epoch: int = 20) -> None:
    filtered = [row for row in history if int(row["epoch"]) >= start_epoch]
    sampled = [row for row in filtered if (int(row["epoch"]) - start_epoch) % 10 == 0 or int(row["epoch"]) in {32, 800}]

    width, height = 900, 520
    left, right, top, bottom = 75, 30, 45, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    epochs = [int(row["epoch"]) for row in filtered]
    train = [float(row["train_loss"]) for row in filtered]
    test = [float(row["test_loss"]) for row in filtered]
    y_max = max(train + test)

    def point(epoch: int, loss: float) -> tuple[float, float]:
        x = left + (epoch - epochs[0]) / max(epochs[-1] - epochs[0], 1) * plot_w
        y = top + (1.0 - loss / max(y_max, 1e-12)) * plot_h
        return x, y

    circles = []
    for row in sampled:
        epoch = int(row["epoch"])
        tx, ty = point(epoch, float(row["train_loss"]))
        vx, vy = point(epoch, float(row["test_loss"]))
        circles.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="2.2" fill="#2563eb"/>')
        circles.append(f'<circle cx="{vx:.2f}" cy="{vy:.2f}" r="2.2" fill="#dc2626"/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="26" text-anchor="middle" font-family="Arial" font-size="18">Raw Loss Points after Warm-up (sampled every 10 epochs)</text>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
  {chr(10).join(circles)}
  <text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">Epoch</text>
  <text x="24" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24 {height / 2})">MSE Loss</text>
  <rect x="{width - 215}" y="54" width="175" height="58" fill="white" stroke="#e5e7eb"/>
  <circle cx="{width - 195}" cy="77" r="4" fill="#2563eb"/>
  <text x="{width - 180}" y="82" font-family="Arial" font-size="13">train loss</text>
  <circle cx="{width - 195}" cy="100" r="4" fill="#dc2626"/>
  <text x="{width - 180}" y="105" font-family="Arial" font-size="13">test loss</text>
  <text x="{left - 10}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="12">{y_max:.4f}</text>
  <text x="{left - 10}" y="{top + plot_h + 4}" text-anchor="end" font-family="Arial" font-size="12">0</text>
</svg>
"""
    write_text(path, svg)


def make_dashboard(metrics: list[dict[str, str]], path: Path) -> None:
    rows = "\n".join(
        f"<tr><td>{row['model']}</td><td>{row['test_mse']}</td><td>{row['test_rmse']}</td><td>{row['test_mae']}</td></tr>"
        for row in metrics
    )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Small Portion Experiment Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    h1 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 28px; max-width: 980px; }}
    img {{ width: 100%; border: 1px solid #e5e7eb; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>10% 데이터 기반 실험 과정 시각화</h1>
  <p>W&B를 직접 사용하지 않고도 발표에 넣을 수 있도록 같은 내용을 로컬 그래프로 정리한 대시보드입니다.</p>
  <div class="grid">
    <section>
      <h2>Epoch별 Train/Test Loss</h2>
      <img src="loss_curve.svg" alt="loss curve">
    </section>
    <section>
      <h2>Overfitting 관측 요약</h2>
      <img src="overfitting_summary.svg" alt="overfitting summary">
    </section>
    <section>
      <h2>Baseline 비교</h2>
      <img src="metrics_bar.svg" alt="metrics bar">
    </section>
    <section>
      <h2>성능 표</h2>
      <table>
        <thead><tr><th>Model</th><th>Test MSE</th><th>Test RMSE</th><th>Test MAE</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    write_text(path, html)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create experiment visualization files.")
    parser.add_argument("--root", type=Path, default=Path("data/experiments/small_portion"))
    args = parser.parse_args()

    root = args.root
    metrics = read_csv(root / "metrics.csv")
    history = read_csv(root / "loss_history.csv")
    make_metric_bar_svg(metrics, root / "metrics_bar.svg")
    make_overfitting_detail_svg(history, root / "overfitting_summary.svg")
    make_loss_curve_svg(history, root / "loss_curve_zoom_epoch20.svg", "Train/Test Loss after Warm-up (epoch 20+)", start_epoch=20)
    make_loss_curve_svg(history, root / "loss_curve_log.svg", "Train/Test Loss (log scale)", log_scale=True)
    make_loss_points_svg(history, root / "loss_points_epoch20.svg")
    make_dashboard(metrics, root / "experiment_dashboard.html")
    print(f"[OK] metrics bar: {(root / 'metrics_bar.svg').resolve()}")
    print(f"[OK] overfitting summary: {(root / 'overfitting_summary.svg').resolve()}")
    print(f"[OK] zoomed loss curve: {(root / 'loss_curve_zoom_epoch20.svg').resolve()}")
    print(f"[OK] log loss curve: {(root / 'loss_curve_log.svg').resolve()}")
    print(f"[OK] raw loss points: {(root / 'loss_points_epoch20.svg').resolve()}")
    print(f"[OK] dashboard: {(root / 'experiment_dashboard.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
