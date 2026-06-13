from __future__ import annotations

"""Pluggable lecture-quality predictors for the Top-K recommender.

The recommendation score blends a preference fit with a *predicted quality*
term (the lecture's expected normalized rating). This module lets that quality
term be produced by different models so the recommender can be compared across
models — the same comparison the offline CV experiments report (RMSE), but seen
through its effect on the actual Top-K lists.

Everything here is numpy-only (no scikit-learn / torch needed at recommend
time). KoBERT models reuse the *precomputed* frozen embeddings in
``data/model/lecture_kobert_embeddings.npz`` (PCA done with a numpy SVD), so no
transformer stack is required to score lectures. sklearn/boosting models can be
added later behind the same ``predict_quality`` interface where those deps
exist; in the CV tables they never beat Ridge anyway.

All predictors fit on all given lectures and predict in-sample — this matches
how the live recommender uses every available review. Out-of-fold CV (for fair
RMSE) lives in the experiment scripts, not here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from topk_engine import TARGET_COLUMN, fit_ridge, predict_ridge, standardize


STRUCTURED_COLUMNS = [
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

QUALITY_MODELS = [
    "ridge_tabular",     # structured 10 + 6 text TF-IDF (current recommender default)
    "ridge_kobert",      # structured 10 + KoBERT PCA32 (best model in the CV study)
    "ridge_structured",  # structured 10 only (ablation)
    "ridge_kobert_only", # KoBERT PCA32 only (ablation)
    "train_mean",        # predict the global mean rating (floor baseline)
    "knn",               # content KNN on tabular features (cosine, mean neighbor rating)
]

DEFAULT_EMBEDDINGS = Path("data/model/lecture_kobert_embeddings.npz")


def _ridge_predict(features: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x, _, _ = standardize(features)
    weights = fit_ridge(x, y.reshape(-1, 1), alpha=alpha)
    return predict_ridge(x, weights)


def _pca(features: np.ndarray, n_components: int) -> np.ndarray:
    """Top-n principal components via SVD on standardized features (numpy only)."""
    centered, _, _ = standardize(features)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return centered @ components.T


def load_kobert_pca(nodes: pd.DataFrame, n_components: int, path: Path = DEFAULT_EMBEDDINGS) -> np.ndarray:
    """Return KoBERT PCA features aligned to ``nodes`` row order. Lectures with
    no embedding get the mean embedding (rare)."""
    data = np.load(path, allow_pickle=True)
    emb_ids = [str(value) for value in data["lecture_id"]]
    embeddings = np.asarray(data["embeddings"], dtype=float)
    by_id = {lecture_id: embeddings[i] for i, lecture_id in enumerate(emb_ids)}
    mean_vec = embeddings.mean(axis=0)
    aligned = np.vstack([by_id.get(str(lid), mean_vec) for lid in nodes["lecture_id"].astype(str)])
    return _pca(aligned, n_components)


def _knn_quality(features: np.ndarray, y: np.ndarray, k: int = 20) -> np.ndarray:
    """Cosine-KNN regression: each lecture's predicted quality = mean rating of
    its k nearest neighbors (excluding itself) in z-scored feature space."""
    x, _, _ = standardize(features)
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    unit = x / norms
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1)[:, :k]
    return y[neighbors].mean(axis=1)


def predict_quality(
    model: str,
    nodes: pd.DataFrame,
    alpha: float = 10.0,
    text_columns: list[str] | None = None,
    embeddings_path: Path = DEFAULT_EMBEDDINGS,
) -> np.ndarray:
    """Predicted normalized quality (length = len(nodes), in [0, 1]) for a model.

    text_columns: the TF-IDF columns to use for ridge_tabular (the recommender
    passes its 6 persona text columns). Defaults to none beyond structured.
    """
    if model not in QUALITY_MODELS:
        raise ValueError(f"Unknown quality model: {model!r}. Choose from {QUALITY_MODELS}.")

    y = nodes[TARGET_COLUMN].to_numpy(dtype=float)
    structured = nodes[STRUCTURED_COLUMNS].to_numpy(dtype=float)
    text_columns = text_columns or []

    if model == "train_mean":
        return np.full(len(nodes), float(y.mean()))

    if model == "ridge_structured":
        return _ridge_predict(structured, y, alpha)

    if model == "ridge_tabular":
        text = nodes[text_columns].to_numpy(dtype=float) if text_columns else np.empty((len(nodes), 0))
        return _ridge_predict(np.hstack([structured, text]), y, alpha)

    if model == "knn":
        text = nodes[text_columns].to_numpy(dtype=float) if text_columns else np.empty((len(nodes), 0))
        return _knn_quality(np.hstack([structured, text]), y)

    if model == "ridge_kobert":
        kobert = load_kobert_pca(nodes, n_components=32, path=embeddings_path)
        return _ridge_predict(np.hstack([structured, kobert]), y, alpha)

    if model == "ridge_kobert_only":
        kobert = load_kobert_pca(nodes, n_components=32, path=embeddings_path)
        return _ridge_predict(kobert, y, alpha)

    raise AssertionError(model)  # unreachable
