from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

PRESETS: dict[str, dict[str, float]] = {
    "low_workload": {
        "assignment_low_score": 1.0,
        "teamwork_low_score": 1.0,
        "grading_generous_score": 0.9,
        "attendance_light_score": 0.6,
        "exam_light_score": 0.9,
        "text_positive_ratio": 0.5,
    },
    "learning_quality": {
        "text_teaching_tfidf": 1.0,
        "text_teaching_positive_tfidf": 1.0,
        "text_positive_ratio": 0.8,
        "grading_generous_score": 0.3,
    },
    "exam_light": {
        "exam_light_score": 1.0,
        "assignment_low_score": 0.6,
        "teamwork_low_score": 0.5,
        "grading_generous_score": 0.7,
        "text_exam_positive_tfidf": 0.4,
    },
    "no_team_project": {
        "teamwork_low_score": 1.0,
        "assignment_low_score": 0.7,
        "exam_light_score": 0.5,
        "grading_generous_score": 0.5,
        "text_positive_ratio": 0.5,
    },
    "challenging_but_good": {
        "assignment_high_score": 0.6,
        "exam_heavy_score": 0.6,
        "text_teaching_positive_tfidf": 1.0,
        "text_teaching_tfidf": 0.7,
        "text_positive_ratio": 0.8,
    },
}


def infer_feature_columns(nodes: pd.DataFrame) -> list[str]:
    text_feature_columns = [
        column
        for column in nodes.columns
        if column.startswith("text_")
        and column not in {"text_review_count", "text_positive_count", "text_negative_count", "text_neutral_count"}
    ]
    return [column for column in BASE_FEATURE_COLUMNS if column in nodes.columns] + text_feature_columns


def load_nodes(path: Path) -> tuple[pd.DataFrame, list[str]]:
    nodes = pd.read_csv(path)
    feature_columns = infer_feature_columns(nodes)
    required = {"lecture_id", TARGET_COLUMN, *feature_columns}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    nodes = nodes.dropna(subset=[TARGET_COLUMN, *feature_columns]).copy()
    nodes["lecture_id"] = nodes["lecture_id"].astype(str)
    nodes[TARGET_COLUMN] = nodes[TARGET_COLUMN].astype(float)
    for column in feature_columns:
        nodes[column] = nodes[column].astype(float)
    return nodes.reset_index(drop=True), feature_columns


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (x - mean) / std, mean, std


def cosine_to_vector(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=1)
    vector_norm = float(np.linalg.norm(vector))
    denominator = np.maximum(x_norm * vector_norm, 1e-12)
    return (x @ vector.reshape(-1)) / denominator


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_with_bias = np.c_[np.ones((len(x), 1)), x]
    identity = np.eye(x_with_bias.shape[1])
    identity[0, 0] = 0.0
    return np.linalg.pinv(x_with_bias.T @ x_with_bias + alpha * identity) @ x_with_bias.T @ y


def predict_ridge(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_with_bias = np.c_[np.ones((len(x), 1)), x]
    return np.clip(x_with_bias @ weights, 0.0, 1.0).reshape(-1)


def load_keywords(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["lecture_id", "top_tfidf_terms"])
    keywords = pd.read_csv(path)
    keywords["lecture_id"] = keywords["lecture_id"].astype(str)
    return keywords[["lecture_id", "top_tfidf_terms"]]


def load_review_examples(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["lecture_id", "sample_review"])
    articles = pd.read_csv(path)
    if "text" not in articles.columns:
        return pd.DataFrame(columns=["lecture_id", "sample_review"])
    articles["lecture_id"] = articles["lecture_id"].astype(str)
    articles["rate"] = pd.to_numeric(articles.get("rate", 0), errors="coerce").fillna(0)
    articles["posvote"] = pd.to_numeric(articles.get("posvote", 0), errors="coerce").fillna(0)
    articles["text"] = articles["text"].fillna("").astype(str)
    articles = articles[articles["text"].str.len() > 0].copy()
    articles = articles.sort_values(["lecture_id", "rate", "posvote"], ascending=[True, False, False])
    examples = articles.groupby("lecture_id", as_index=False).first()[["lecture_id", "text"]]
    examples["sample_review"] = examples["text"].map(lambda text: text[:120].replace("\n", " "))
    return examples[["lecture_id", "sample_review"]]


def preference_from_args(args: argparse.Namespace, feature_columns: list[str]) -> tuple[str, dict[str, float]]:
    if args.weights_json:
        weights = json.loads(args.weights_json)
        if not isinstance(weights, dict):
            raise ValueError("--weights-json must be a JSON object.")
        return "custom", {str(key): float(value) for key, value in weights.items()}

    if args.weights_file:
        with args.weights_file.open("r", encoding="utf-8") as file:
            weights = json.load(file)
        if not isinstance(weights, dict):
            raise ValueError("--weights-file must contain a JSON object.")
        return args.weights_file.stem, {str(key): float(value) for key, value in weights.items()}

    if args.preset not in PRESETS:
        raise ValueError(f"Unknown preset: {args.preset}")
    return args.preset, PRESETS[args.preset]


def build_preference_vector(weights: dict[str, float], feature_columns: list[str]) -> np.ndarray:
    unknown = sorted(set(weights) - set(feature_columns))
    if unknown:
        raise ValueError(f"Unknown feature(s) in preference weights: {unknown}")
    vector = np.zeros(len(feature_columns), dtype=float)
    for idx, column in enumerate(feature_columns):
        vector[idx] = float(weights.get(column, 0.0))
    if math.isclose(float(np.linalg.norm(vector)), 0.0):
        raise ValueError("Preference vector is empty. Give at least one positive feature weight.")
    return vector


def make_recommendations(
    nodes: pd.DataFrame,
    feature_columns: list[str],
    preference_weights: dict[str, float],
    similarity_weight: float,
    quality_weight: float,
    alpha: float,
) -> pd.DataFrame:
    x_raw = nodes[feature_columns].to_numpy(dtype=float)
    x, _, _ = standardize(x_raw)
    y = nodes[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)
    ridge_weights = fit_ridge(x, y, alpha=alpha)
    predicted_quality = predict_ridge(x, ridge_weights)

    preference_raw = build_preference_vector(preference_weights, feature_columns)
    preference_standardized = preference_raw / np.maximum(x_raw.std(axis=0), 1e-12)
    similarity = cosine_to_vector(x, preference_standardized)
    similarity_01 = (similarity + 1.0) / 2.0

    score = similarity_weight * similarity_01 + quality_weight * predicted_quality
    result = nodes[
        [
            "lecture_id",
            "article_count",
            "rating_count",
            "rating_average",
            TARGET_COLUMN,
            *feature_columns,
        ]
    ].copy()
    result.insert(1, "recommendation_score", score)
    result.insert(2, "preference_similarity", similarity_01)
    result.insert(3, "predicted_quality_norm", predicted_quality)
    result.insert(4, "predicted_quality_5pt", predicted_quality * 5.0)
    result = result.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def write_summary(
    path: Path,
    scenario_name: str,
    preference_weights: dict[str, float],
    recommendations: pd.DataFrame,
    similarity_weight: float,
    quality_weight: float,
    alpha: float,
) -> None:
    top = recommendations.head(10)
    lines = [
        "# Top-K Recommendation Summary",
        "",
        f"- scenario: `{scenario_name}`",
        f"- score: `{similarity_weight:.2f} * preference_similarity + {quality_weight:.2f} * predicted_quality`",
        f"- quality model: Ridge regression, alpha={alpha}",
        "- limitation: current raw data has lecture_id only, so course/professor names must be merged later.",
        "",
        "## Preference Weights",
        "",
    ]
    for feature, weight in sorted(preference_weights.items()):
        lines.append(f"- `{feature}`: {weight}")

    lines.extend(["", "## Top 10", ""])
    for _, row in top.iterrows():
        keywords = str(row.get("top_tfidf_terms", "") or "")
        keyword_text = keywords[:90]
        lines.append(
            f"{int(row['rank'])}. lecture_id `{row['lecture_id']}` | "
            f"score {row['recommendation_score']:.4f} | "
            f"similarity {row['preference_similarity']:.4f} | "
            f"predicted {row['predicted_quality_5pt']:.2f}/5 | "
            f"actual avg {row['rating_average']:.2f}/5 | "
            f"keywords: {keyword_text}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank lectures by a user preference vector and predicted lecture quality.")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--keywords", type=Path, default=Path("data/model/lecture_top_keywords.csv"))
    parser.add_argument("--articles", type=Path, default=Path("data/normalized/lecture_articles.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/recommendations"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="low_workload")
    parser.add_argument("--weights-json", default="")
    parser.add_argument("--weights-file", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--similarity-weight", type=float, default=0.7)
    parser.add_argument("--quality-weight", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isclose(args.similarity_weight + args.quality_weight, 1.0, abs_tol=1e-8):
        raise ValueError("--similarity-weight and --quality-weight must sum to 1.0")

    nodes, feature_columns = load_nodes(args.nodes)
    scenario_name, preference_weights = preference_from_args(args, feature_columns)
    recommendations = make_recommendations(
        nodes=nodes,
        feature_columns=feature_columns,
        preference_weights=preference_weights,
        similarity_weight=args.similarity_weight,
        quality_weight=args.quality_weight,
        alpha=args.alpha,
    )

    keywords = load_keywords(args.keywords)
    examples = load_review_examples(args.articles)
    recommendations = recommendations.merge(keywords, on="lecture_id", how="left")
    recommendations = recommendations.merge(examples, on="lecture_id", how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = scenario_name.replace(" ", "_")
    all_path = args.out_dir / f"topk_{stem}_all.csv"
    top_path = args.out_dir / f"topk_{stem}_{args.top_k}.csv"
    summary_path = args.out_dir / f"topk_{stem}_summary.md"

    recommendations.to_csv(all_path, index=False, encoding="utf-8-sig")
    recommendations.head(args.top_k).to_csv(top_path, index=False, encoding="utf-8-sig")
    write_summary(
        summary_path,
        scenario_name,
        preference_weights,
        recommendations,
        args.similarity_weight,
        args.quality_weight,
        args.alpha,
    )

    print(f"[OK] Wrote full ranking: {all_path}")
    print(f"[OK] Wrote top-{args.top_k}: {top_path}")
    print(f"[OK] Wrote summary: {summary_path}")
    print("")
    print(recommendations[["rank", "lecture_id", "recommendation_score", "preference_similarity", "predicted_quality_5pt", "rating_average"]].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
