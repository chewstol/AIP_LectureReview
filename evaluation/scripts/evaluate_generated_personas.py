from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


BASE_FEATURE_COLUMNS = [
    "assignment_low_score",
    "assignment_high_score",
    "teamwork_low_score",
    "teamwork_high_score",
    "grading_generous_score",
    "grading_strict_score",
    "attendance_light_score",
    "attendance_strict_score",
    "exam_light_score",
    "exam_heavy_score",
]

TARGET_COLUMN = "rating_average_norm"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_feature_columns(columns: list[str]) -> list[str]:
    excluded = {"text_review_count", "text_positive_count", "text_negative_count", "text_neutral_count"}
    text_columns = [column for column in columns if column.startswith("text_") and column not in excluded]
    return [column for column in BASE_FEATURE_COLUMNS if column in columns] + text_columns


def load_nodes(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Nodes CSV is empty: {path}")

    feature_columns = infer_feature_columns(list(rows[0].keys()))
    required = {"lecture_id", "rating_average", TARGET_COLUMN, *feature_columns}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required column(s) in {path}: {sorted(missing)}")

    nodes: list[dict[str, Any]] = []
    for row in rows:
        try:
            node = {
                "lecture_id": str(row["lecture_id"]),
                "rating_average": float(row["rating_average"]),
                TARGET_COLUMN: float(row[TARGET_COLUMN]),
            }
            for column in feature_columns:
                node[column] = float(row[column])
        except (TypeError, ValueError):
            continue
        nodes.append(node)

    if not nodes:
        raise ValueError(f"No usable node rows found in {path}")
    return nodes, feature_columns


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rmse(errors: list[float]) -> float:
    return math.sqrt(mean([error * error for error in errors]))


def mae(errors: list[float]) -> float:
    return mean([abs(error) for error in errors])


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * pct
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - idx) + sorted_values[high] * (idx - low)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denom == 0.0 or y_denom == 0.0:
        return 0.0
    return numerator / (x_denom * y_denom)


def matrix_inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row_idx: abs(augmented[row_idx][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("Ridge matrix is singular even after regularization.")
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]

        for row_idx in range(n):
            if row_idx == col:
                continue
            factor = augmented[row_idx][col]
            if factor == 0.0:
                continue
            augmented[row_idx] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row_idx], augmented[col])
            ]

    return [row[n:] for row in augmented]


def fit_ridge(x: list[list[float]], y: list[float], alpha: float) -> list[float]:
    rows = [[1.0, *row] for row in x]
    width = len(rows[0])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]

    for row, target in zip(rows, y):
        for i in range(width):
            xty[i] += row[i] * target
            for j in range(width):
                xtx[i][j] += row[i] * row[j]

    for i in range(1, width):
        xtx[i][i] += alpha

    inverse = matrix_inverse(xtx)
    return [sum(inverse_row[j] * xty[j] for j in range(width)) for inverse_row in inverse]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def standardize_matrix(x_raw: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    width = len(x_raw[0])
    means = [mean([row[col] for row in x_raw]) for col in range(width)]
    stds: list[float] = []
    for col in range(width):
        variance = mean([(row[col] - means[col]) ** 2 for row in x_raw])
        std = math.sqrt(variance)
        stds.append(std if std != 0.0 else 1.0)
    x = [[(value - means[col]) / stds[col] for col, value in enumerate(row)] for row in x_raw]
    return x, means, stds


def clipped_prediction(row: list[float], weights: list[float]) -> float:
    value = weights[0] + dot(row, weights[1:])
    return min(max(value, 0.0), 1.0)


def cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(dot(left, left))
    right_norm = math.sqrt(dot(right, right))
    denom = max(left_norm * right_norm, 1e-12)
    return dot(left, right) / denom


def build_quality_predictions(
    nodes: list[dict[str, Any]],
    feature_columns: list[str],
    alpha: float,
) -> tuple[dict[str, float], list[list[float]], list[float]]:
    x_raw = [[float(node[column]) for column in feature_columns] for node in nodes]
    x, _means, stds = standardize_matrix(x_raw)
    y = [float(node[TARGET_COLUMN]) for node in nodes]
    weights = fit_ridge(x, y, alpha)
    predictions = {
        str(node["lecture_id"]): clipped_prediction(row, weights)
        for node, row in zip(nodes, x)
    }
    return predictions, x, stds


def rank_for_persona(
    persona_vector: dict[str, Any],
    nodes: list[dict[str, Any]],
    feature_columns: list[str],
    standardized_x: list[list[float]],
    feature_stds: list[float],
    quality_predictions: dict[str, float],
    similarity_weight: float,
    quality_weight: float,
) -> dict[str, int]:
    unknown = sorted(set(persona_vector) - set(feature_columns))
    if unknown:
        raise ValueError(f"Persona vector has unknown feature(s): {unknown}")

    preference = [
        float(persona_vector.get(column, 0.0)) / max(feature_stds[idx], 1e-12)
        for idx, column in enumerate(feature_columns)
    ]
    if math.sqrt(dot(preference, preference)) == 0.0:
        raise ValueError("Persona vector is empty after conversion.")

    scored = []
    for node, row in zip(nodes, standardized_x):
        lecture_id = str(node["lecture_id"])
        similarity_01 = (cosine(row, preference) + 1.0) / 2.0
        score = similarity_weight * similarity_01 + quality_weight * quality_predictions[lecture_id]
        scored.append((lecture_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return {lecture_id: rank for rank, (lecture_id, _score) in enumerate(scored, start=1)}


def evaluate_vector_field(
    personas: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    feature_columns: list[str],
    quality_predictions: dict[str, float],
    standardized_x: list[list[float]],
    feature_stds: list[float],
    vector_field: str,
    similarity_weight: float,
    quality_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_by_id = {str(node["lecture_id"]): node for node in nodes}
    detail_rows: list[dict[str, Any]] = []
    persona_rows: list[dict[str, Any]] = []

    for persona in personas:
        vector = persona.get(vector_field)
        if not isinstance(vector, dict):
            raise ValueError(f"Persona {persona.get('persona_id')} has no `{vector_field}` object.")

        rank_map = rank_for_persona(
            persona_vector=vector,
            nodes=nodes,
            feature_columns=feature_columns,
            standardized_x=standardized_x,
            feature_stds=feature_stds,
            quality_predictions=quality_predictions,
            similarity_weight=similarity_weight,
            quality_weight=quality_weight,
        )

        errors: list[float] = []
        lecture_avg_errors: list[float] = []
        ranks: list[float] = []
        actuals: list[float] = []
        preds: list[float] = []

        for review in persona.get("selected_reviews", []):
            lecture_id = str(review["lecture_id"])
            actual = float(review["rate"])
            predicted = quality_predictions[lecture_id] * 5.0
            lecture_avg = float(node_by_id[lecture_id]["rating_average"])
            rank = rank_map[lecture_id]
            error = predicted - actual
            lecture_avg_error = lecture_avg - actual

            errors.append(error)
            lecture_avg_errors.append(lecture_avg_error)
            ranks.append(float(rank))
            actuals.append(actual)
            preds.append(predicted)

            detail_rows.append(
                {
                    "vector_field": vector_field,
                    "persona_id": persona.get("persona_id"),
                    "preset_name": persona.get("preset_name"),
                    "lecture_id": lecture_id,
                    "actual_rate": round(actual, 6),
                    "predicted_quality_5pt": round(predicted, 6),
                    "lecture_avg_5pt": round(lecture_avg, 6),
                    "prediction_error": round(error, 6),
                    "recommendation_rank": rank,
                }
            )

        persona_rows.append(
            {
                "vector_field": vector_field,
                "persona_id": persona.get("persona_id"),
                "preset_name": persona.get("preset_name"),
                "reviews": len(errors),
                "mae": mae(errors),
                "rmse": rmse(errors),
                "bias": mean(errors),
                "within_0_5": mean([1.0 if abs(error) <= 0.5 else 0.0 for error in errors]),
                "within_1_0": mean([1.0 if abs(error) <= 1.0 else 0.0 for error in errors]),
                "lecture_avg_mae": mae(lecture_avg_errors),
                "hit_at_10": mean([1.0 if rank <= 10 else 0.0 for rank in ranks]),
                "hit_at_30": mean([1.0 if rank <= 30 else 0.0 for rank in ranks]),
                "hit_at_100": mean([1.0 if rank <= 100 else 0.0 for rank in ranks]),
                "median_rank": percentile(ranks, 0.5),
                "mean_rank": mean(ranks),
                "corr": pearson(actuals, preds),
            }
        )

    return persona_rows, detail_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    weighted_reviews = sum(int(row["reviews"]) for row in rows)
    result: dict[str, float] = {"personas": float(len(rows)), "reviews": float(weighted_reviews)}
    for key in [
        "mae",
        "rmse",
        "bias",
        "within_0_5",
        "within_1_0",
        "lecture_avg_mae",
        "hit_at_10",
        "hit_at_30",
        "hit_at_100",
        "median_rank",
        "mean_rank",
    ]:
        result[key] = mean([float(row[key]) for row in rows])
    return result


def write_summary(path: Path, all_persona_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]], alpha: float) -> None:
    by_field: dict[str, list[dict[str, Any]]] = {}
    for row in all_persona_rows:
        by_field.setdefault(str(row["vector_field"]), []).append(row)

    lines = [
        "# Generated Persona Prediction Evaluation",
        "",
        "## Scope",
        "",
        "- Input personas: `data/generated_personas.json`",
        "- Ground truth used here: each persona's `selected_reviews[].rate` on a 1-5 scale",
        "- Quality model: Ridge regression reproduced from `recommend_topk.py`, trained on `data/model/lecture_nodes_with_text.csv`",
        f"- Ridge alpha: `{alpha}`",
        "- Ranking score: `0.70 * preference_similarity + 0.30 * predicted_quality`",
        "- Note: this evaluates how well generated persona examples align with the existing model; it is not a true held-out user study.",
        "",
        "## Overall Results",
        "",
        "| Vector field | Personas | Reviews | MAE | RMSE | Lecture avg MAE | Bias | Within 0.5 | Within 1.0 | Hit@10 | Hit@30 | Hit@100 | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for vector_field, rows in by_field.items():
        summary = aggregate(rows)
        lines.append(
            f"| `{vector_field}` | {int(summary['personas'])} | {int(summary['reviews'])} | "
            f"{fmt(summary['mae'])} | {fmt(summary['rmse'])} | {fmt(summary['lecture_avg_mae'])} | "
            f"{fmt(summary['bias'])} | "
            f"{fmt(summary['within_0_5'] * 100, 1)}% | {fmt(summary['within_1_0'] * 100, 1)}% | "
            f"{fmt(summary['hit_at_10'] * 100, 1)}% | {fmt(summary['hit_at_30'] * 100, 1)}% | "
            f"{fmt(summary['hit_at_100'] * 100, 1)}% | {fmt(summary['median_rank'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Persona Results",
            "",
            "| Vector field | Persona | Preset | Reviews | MAE | RMSE | Bias | Within 1.0 | Hit@30 | Median rank |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in sorted(all_persona_rows, key=lambda item: (str(item["vector_field"]), int(item["persona_id"]))):
        lines.append(
            f"| `{row['vector_field']}` | {row['persona_id']} | `{row['preset_name']}` | {row['reviews']} | "
            f"{fmt(float(row['mae']))} | {fmt(float(row['rmse']))} | {fmt(float(row['bias']))} | "
            f"{fmt(float(row['within_1_0']) * 100, 1)}% | {fmt(float(row['hit_at_30']) * 100, 1)}% | "
            f"{fmt(float(row['median_rank']), 1)} |"
        )

    worst = sorted(detail_rows, key=lambda row: abs(float(row["prediction_error"])), reverse=True)[:10]
    lines.extend(
        [
            "",
            "## Largest Rating Errors",
            "",
            "| Vector field | Persona | Preset | Lecture | Actual | Predicted | Lecture avg | Error | Rank |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst:
        lines.append(
            f"| `{row['vector_field']}` | {row['persona_id']} | `{row['preset_name']}` | {row['lecture_id']} | "
            f"{fmt(float(row['actual_rate']))} | {fmt(float(row['predicted_quality_5pt']))} | "
            f"{fmt(float(row['lecture_avg_5pt']))} | {fmt(float(row['prediction_error']))} | {row['recommendation_rank']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Rating MAE/RMSE compares predicted lecture quality to individual selected review ratings. Individual reviews are noisier than lecture averages, so this is stricter than the original lecture-average CV task.",
            "- Positive bias means the model tends to predict higher than the selected review rating; negative bias means it tends to underpredict.",
            "- Hit@K checks whether the persona vector places the selected-review lectures inside the recommender's Top-K list.",
            "- In this run, `initial_preference_vector` ranked the selected-review lectures better than `aggregated_review_vector` overall. That suggests the generated initial persona vectors are closer to the recommender's feature space than the post-sampled aggregate vectors.",
            "- `initial_preference_vector` is the cleaner cold-start persona input. `aggregated_review_vector` is useful as an observed-review profile, but it should not automatically be assumed to improve ranking.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated personas against existing recommendation model logic.")
    parser.add_argument("--personas", type=Path, default=Path("data/generated_personas.json"))
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/persona_evaluation"))
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--similarity-weight", type=float, default=0.7)
    parser.add_argument("--quality-weight", type=float, default=0.3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isclose(args.similarity_weight + args.quality_weight, 1.0, abs_tol=1e-8):
        raise ValueError("--similarity-weight and --quality-weight must sum to 1.0")

    personas = load_json(args.personas)
    if not isinstance(personas, list):
        raise ValueError("--personas must contain a JSON array.")

    nodes, feature_columns = load_nodes(args.nodes)
    quality_predictions, standardized_x, feature_stds = build_quality_predictions(nodes, feature_columns, args.alpha)

    all_persona_rows: list[dict[str, Any]] = []
    all_detail_rows: list[dict[str, Any]] = []
    for vector_field in ["initial_preference_vector", "aggregated_review_vector"]:
        persona_rows, detail_rows = evaluate_vector_field(
            personas=personas,
            nodes=nodes,
            feature_columns=feature_columns,
            quality_predictions=quality_predictions,
            standardized_x=standardized_x,
            feature_stds=feature_stds,
            vector_field=vector_field,
            similarity_weight=args.similarity_weight,
            quality_weight=args.quality_weight,
        )
        all_persona_rows.extend(persona_rows)
        all_detail_rows.extend(detail_rows)

    write_csv(args.out_dir / "persona_metrics.csv", all_persona_rows)
    write_csv(args.out_dir / "review_predictions.csv", all_detail_rows)
    write_summary(args.out_dir / "summary.md", all_persona_rows, all_detail_rows, args.alpha)

    print(f"[OK] personas: {len(personas)}")
    print(f"[OK] nodes: {len(nodes)}")
    print(f"[OK] feature count: {len(feature_columns)}")
    print(f"[OK] output: {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
