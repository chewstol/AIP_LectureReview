from __future__ import annotations

import math

import numpy as np
import pandas as pd


TARGET_COLUMN = "rating_average_norm"


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


def select_topk(recommendations: pd.DataFrame, k: int) -> pd.DataFrame:
    return recommendations.head(k).copy()
