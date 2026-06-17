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


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] by rank (ties broken arbitrarily). Makes the
    blend weights comparable when two score components have different spreads."""
    order = np.argsort(np.argsort(values))
    return order.astype(float) / max(len(values) - 1, 1)


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
        # Unspecified features stay 0. Because the lecture matrix is z-scored
        # (per-column mean 0), 0 here means "feature average / neutral", which
        # is exactly the desired default — no separate mean-fill needed.
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
    shrinkage_m: float = 0.0,
    normalize_components: bool = False,
) -> pd.DataFrame:
    x_raw = nodes[feature_columns].to_numpy(dtype=float)
    x, _, _ = standardize(x_raw)
    y = nodes[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)
    ridge_weights = fit_ridge(x, y, alpha=alpha)
    predicted_quality = predict_ridge(x, ridge_weights)

    if shrinkage_m > 0.0:
        # Empirical-Bayes shrinkage toward the global mean rating. Lectures with
        # few reviews get pulled to the prior so a high rating from 1-2 reviews
        # does not outrank a well-reviewed lecture.
        prior = float(y.mean())
        rating_count = np.nan_to_num(nodes["rating_count"].to_numpy(dtype=float), nan=0.0)
        predicted_quality = (rating_count * predicted_quality + shrinkage_m * prior) / (rating_count + shrinkage_m)

    # Preference weights are a user-specified direction, not a sample, so they are
    # used as-is against the z-scored lecture matrix. The weight ratios between
    # features are preserved (no per-column std rescaling, no mean subtraction).
    preference_vector = build_preference_vector(preference_weights, feature_columns)
    similarity = cosine_to_vector(x, preference_vector)
    similarity_01 = (similarity + 1.0) / 2.0

    if normalize_components:
        # Rank-normalize each component so similarity_weight / quality_weight
        # reflect the intended blend regardless of each component's spread.
        sim_component = rank_normalize(similarity_01)
        quality_component = rank_normalize(predicted_quality)
    else:
        sim_component = similarity_01
        quality_component = predicted_quality

    score = similarity_weight * sim_component + quality_weight * quality_component
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


def select_topk(
    recommendations: pd.DataFrame,
    k: int,
    feature_columns: list[str] | None = None,
    diversity: float = 1.0,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """Pick the top-k rows.

    min_similarity: drop candidates whose preference_similarity is below this.
    diversity (λ): 1.0 = pure score (head-k). Below 1.0 applies MMR so the list
    avoids near-duplicate lectures. Requires feature_columns to measure overlap.
    """
    filtered = recommendations[recommendations["preference_similarity"] >= min_similarity].copy()
    filtered = filtered.reset_index(drop=True)

    if diversity >= 1.0 or feature_columns is None or len(filtered) <= k:
        return filtered.head(k).copy()

    # Overlap is measured on the z-scored feature matrix to stay consistent with
    # how the recommendation score treats features.
    feats, _, _ = standardize(filtered[feature_columns].to_numpy(dtype=float))

    relevance = filtered["recommendation_score"].to_numpy(dtype=float)
    rel_min, rel_max = float(relevance.min()), float(relevance.max())
    relevance_norm = (relevance - rel_min) / max(rel_max - rel_min, 1e-12)

    lam = diversity
    selected: list[int] = []
    candidates = list(range(len(filtered)))
    while candidates and len(selected) < k:
        if not selected:
            best = max(candidates, key=lambda i: relevance_norm[i])
        else:
            selected_matrix = feats[selected]

            def mmr(i: int) -> float:
                sims = cosine_to_vector(selected_matrix, feats[i])
                max_overlap = (float(sims.max()) + 1.0) / 2.0
                return lam * relevance_norm[i] - (1.0 - lam) * max_overlap

            best = max(candidates, key=mmr)
        selected.append(best)
        candidates.remove(best)

    return filtered.iloc[selected].copy()


def face_validity(full: pd.DataFrame, topk: pd.DataFrame, specified_columns: list[str]) -> float:
    """Mean z-score of the user-specified features across the top-k list.
    Positive => the recommended lectures really are above average on what the
    user asked for. 0 => no better than the population average."""
    if not specified_columns or len(topk) == 0:
        return 0.0
    population = full[specified_columns].to_numpy(dtype=float)
    mean = population.mean(axis=0)
    std = population.std(axis=0)
    std[std == 0.0] = 1.0
    z = (topk[specified_columns].to_numpy(dtype=float) - mean) / std
    return float(z.mean())


def intra_list_similarity(full: pd.DataFrame, topk: pd.DataFrame, feature_columns: list[str]) -> float:
    """Average pairwise cosine similarity within the top-k list (z-scored
    features). Lower = more diverse. Used to measure the MMR effect."""
    if len(topk) < 2:
        return 0.0
    population = full[feature_columns].to_numpy(dtype=float)
    mean = population.mean(axis=0)
    std = population.std(axis=0)
    std[std == 0.0] = 1.0
    z = (topk[feature_columns].to_numpy(dtype=float) - mean) / std
    norms = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    unit = z / norms
    sims = unit @ unit.T
    upper = np.triu_indices(len(z), k=1)
    return float(sims[upper].mean())
