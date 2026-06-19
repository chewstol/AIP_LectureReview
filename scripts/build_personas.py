from __future__ import annotations

"""Build persona test vectors from the about-class forum dump.

A *persona* is a synthetic "person" analyzed exactly like a lecture: it is
described by the same 16 feature columns a lecture node uses (10 structured
workload axes + 6 category TF-IDF columns) and is then handed to the Top-K
engine as a preference vector to produce lecture recommendations.

Pipeline (per persona):
  1. Randomly sample ``--sample-size`` posts (each = a forum post + its
     comments) from ``about-class/*.json``.
  2. Ask Ollama to discover the single dominant theme among the sample and
     return the indices of ``--select-min``..``--select-max`` posts that match
     it (option: dominant-theme auto-discovery).
  3. Build a *tendency seed* from the selected posts+comments: the 6 category
     TF-IDF features (normalized onto the lecture corpus scale) and, in
     ``--seed-mode preference``, the 10 structured axes derived from that
     salience + valence.
  4. Retrieve the lectures most similar to the seed (cosine over the z-scored
     lecture matrix) and AVERAGE the 16-dim vectors of a random 10-20 of them.
     This aggregate is taken as one student's persona vector, so the persona
     lives on the manifold of real, already-analyzed lectures.
  5. Write ``persona_{i}.json`` (a recommend_topk --weights-file) + a meta
     file, and optionally run Top-K for that persona.

Only numpy / pandas / stdlib are used (Ollama is called over its HTTP API with
urllib). This is a NEW component; topk_engine.py stays a pure-compute module.
"""

import argparse
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_text_features import (
    CATEGORY_KEYWORDS,
    category_scores,
    compute_idf,
    read_csv,
    tokenize,
)
from recommend_topk import load_keywords, load_review_examples
from topk_engine import (
    TARGET_COLUMN,
    build_preference_vector,
    cosine_to_vector,
    make_recommendations,
    select_topk,
    standardize,
)


# The reduced 16-feature schema the persona is described in. Every one of these
# columns already exists in data/model/lecture_nodes_with_text.csv, so lectures
# and personas are compared on exactly the same axes.
STRUCTURED_FEATURES = [
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

# build_text_features emits text_{category}_tfidf for each CATEGORY_KEYWORDS key.
TEXT_FEATURES = [f"text_{category}_tfidf" for category in CATEGORY_KEYWORDS]

PERSONA_FEATURE_COLUMNS = STRUCTURED_FEATURES + TEXT_FEATURES

# Complementary structured pairs (low/high). Used to sanity-fill the estimate
# when Ollama only gives one side of a pair.
STRUCTURED_PAIRS = [
    ("assignment_low_score", "assignment_high_score"),
    ("teamwork_low_score", "teamwork_high_score"),
    ("grading_generous_score", "grading_strict_score"),
    ("attendance_light_score", "attendance_strict_score"),
    ("exam_light_score", "exam_heavy_score"),
]


# --------------------------------------------------------------------------- #
# about-class loading
# --------------------------------------------------------------------------- #
def load_about_class(about_dir: Path) -> list[dict[str, Any]]:
    """Read every about-class json into post records.

    Each record holds the article (title + body) AND every comment, because the
    discussion under a post often carries the real signal (e.g. a question post
    whose comments say "테케 빡셈, 보고서 정성들여 써도 1점"). Comments feed both
    the theme-selection prompt and the TF-IDF / structured-score analysis.

    ``snippet``   = title + body, for compact display.
    ``full_text`` = title + body + all comment texts, for analysis + selection.
    ``categories``= which CATEGORY_KEYWORDS axes the post touches (for sampling).
    """
    posts: list[dict[str, Any]] = []
    for path in sorted(about_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        article = data.get("article") or {}
        title = str(article.get("title", "") or "").strip()
        body = str(article.get("text", "") or "").strip()
        comments = [
            str(comment.get("text", "") or "").strip()
            for comment in (data.get("comments") or [])
        ]
        comments = [text for text in comments if text]
        snippet = (title + " " + body).strip()
        full_text = " ".join([snippet, *comments]).strip()
        if not full_text:
            continue
        posts.append(
            {
                "id": str(article.get("id", path.stem)),
                "snippet": snippet,
                "comments": comments,
                "full_text": full_text,
                "comment_count": len(comments),
                "categories": post_categories(full_text),
                "positive": is_seeker_post(full_text),
            }
        )
    return posts


def load_reviews(articles_path: Path, min_length: int = 10) -> list[dict[str, Any]]:
    """Read lecture reviews (data/normalized/lecture_articles.csv) into the SAME
    post-record schema as load_about_class, so personas can be built from the
    lecture-review corpus instead of the about-class forum.

    Each review row is one document: it has no comments, so ``snippet`` ==
    ``full_text`` == the review text. ``positive`` (seeker bias) uses the review's
    star rating (rate >= 4) when available, falling back to the praise lexicon.
    This makes the persona's TF-IDF features come from exactly the same text the
    lecture nodes were built from — handy for explaining 'why TF-IDF'."""
    posts: list[dict[str, Any]] = []
    for row in read_csv(articles_path):
        text = str(row.get("text", "") or "").strip()
        if len(text) < min_length:
            continue
        try:
            rate = float(row.get("rate", "") or "nan")
        except (TypeError, ValueError):
            rate = float("nan")
        positive = (rate >= 4.0) if rate == rate else is_seeker_post(text)  # rate==rate: not NaN
        posts.append(
            {
                "id": str(row.get("article_id") or row.get("lecture_id") or len(posts)),
                "snippet": text,
                "comments": [],
                "full_text": text,
                "comment_count": 0,
                "categories": post_categories(text),
                "positive": positive,
            }
        )
    return posts


# Lexicon for "seeker" posts: praise / worth-it / substantial-class signals,
# deliberately excluding "easy/꿀" cues (those are still burden-averse).
SEEKER_LEXICON = (
    "갓강", "명강", "알차", "유익", "배울", "얻어가", "도움", "추천", "최고",
    "열정", "인생강의", "좋았", "좋아요", "유익했", "재밌었", "갓성비",
)


def is_seeker_post(text: str) -> bool:
    """True if the post praises a class as worth-it / substantial (seeker
    signal), used to bias sampling for challenge/quality-seeking personas."""
    return any(term in text for term in SEEKER_LEXICON)


def post_categories(text: str) -> set[str]:
    """Which workload axes a post mentions, by CATEGORY_KEYWORDS hit. Used to
    bias per-persona sampling toward a target axis for diversity."""
    tokens = set(tokenize(text))
    hits: set[str] = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if tokens & {keyword.lower() for keyword in keywords}:
            hits.add(category)
    return hits


# --------------------------------------------------------------------------- #
# Lecture-corpus TF-IDF scale (so persona text features match lecture nodes)
# --------------------------------------------------------------------------- #
def build_lecture_text_scale(articles_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Replicate build_text_features' pre-normalization step over the lecture
    corpus to recover (idf, per-category max). The persona text features are
    later divided by this max, putting them on the same [0, 1] scale lecture
    nodes were normalized to."""
    articles = read_csv(articles_path)
    tokenized = [tokenize(row.get("text", "")) for row in articles]
    idf = compute_idf(tokenized)

    # Per-lecture aggregated category score = sum over its reviews / review_count
    # (exactly what build_text_features does before normalize_scores).
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for row, tokens in zip(articles, tokenized):
        lecture_id = row["lecture_id"]
        scores = category_scores(tokens, idf)
        bucket = sums.setdefault(lecture_id, {category: 0.0 for category in CATEGORY_KEYWORDS})
        for category, score in scores.items():
            bucket[category] += score
        counts[lecture_id] = counts.get(lecture_id, 0) + 1

    category_max: dict[str, float] = {category: 0.0 for category in CATEGORY_KEYWORDS}
    for lecture_id, bucket in sums.items():
        review_count = max(counts[lecture_id], 1)
        for category in CATEGORY_KEYWORDS:
            category_max[category] = max(category_max[category], bucket[category] / review_count)
    return idf, category_max


def persona_text_features(
    selected_posts: list[dict[str, str]],
    idf: dict[str, float],
    category_max: dict[str, float],
) -> dict[str, float]:
    """Compute the 6 text_{category}_tfidf features for the persona, on the
    lecture corpus scale, by treating each selected post as one review doc."""
    totals = {category: 0.0 for category in CATEGORY_KEYWORDS}
    for post in selected_posts:
        scores = category_scores(tokenize(post["full_text"]), idf)
        for category, score in scores.items():
            totals[category] += score

    review_count = max(len(selected_posts), 1)
    features: dict[str, float] = {}
    for category in CATEGORY_KEYWORDS:
        raw = totals[category] / review_count
        ceiling = category_max.get(category, 0.0)
        value = raw / ceiling if ceiling > 0.0 else 0.0
        features[f"text_{category}_tfidf"] = round(min(value, 1.0), 6)
    return features


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def ollama_json(host: str, model: str, prompt: str, num_predict: int, num_ctx: int = 12288) -> dict[str, Any]:
    """Call Ollama /api/generate with think disabled + JSON format and parse
    the response body as a JSON object.

    num_ctx must hold the WHOLE prompt + the generated answer. The theme-selection
    catalog (100 posts) is ~7-8k tokens, which overflows the model's small default
    context and leaves ~0 room for output (done_reason='length', 1 token) — so we
    raise the window explicitly rather than relying on the server default."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0.2, "num_predict": num_predict, "num_ctx": num_ctx},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = (body.get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return json.loads(text)


CATEGORY_LABELS = {
    "assignment": "과제",
    "exam": "시험",
    "teamwork": "팀플",
    "attendance": "출석",
    "grading": "학점",
    "teaching": "강의력",
}


def select_theme_posts(
    host: str,
    model: str,
    sample: list[dict[str, Any]],
    select_min: int,
    select_max: int,
    target_category: str | None = None,
) -> tuple[str, list[int]]:
    """Ask the model to find the dominant theme and return matching indices.

    The catalog includes each post's comments (truncated), so the model judges
    tendency from the whole discussion, not just the article title/body.
    ``target_category`` softly steers which axis the theme should center on."""
    lines = []
    for index, post in enumerate(sample):
        text = post["full_text"].replace("\n", " ").replace("<br />", " ")[:160]
        lines.append(f"[{index}] {text}")
    catalog = "\n".join(lines)

    if target_category:
        focus = (
            f"이 글들은 주로 '{CATEGORY_LABELS.get(target_category, target_category)}' "
            "측면과 관련이 있습니다. 그 측면을 중심으로, "
        )
    else:
        focus = ""

    prompt = (
        "다음은 대학 강의 게시판 글 목록입니다 (각 줄: [번호] 글 본문과 댓글).\n"
        f"{focus}이 글들 중에서 가장 응집력 있는 '하나의 구체적인 지배적 테마'를 찾으세요. "
        "테마는 과제/시험/팀플/출석/학점/강의력 부담이나 만족도 같은 수강 경험의 한 측면입니다.\n"
        f"그 테마에 잘 맞는 글(댓글 내용 포함)을 {select_min}~{select_max}개 고르세요.\n\n"
        f"{catalog}\n\n"
        '반드시 다음 JSON 형식으로만 답하세요: '
        '{"theme": "한 문장 테마 설명", "selected": [번호, 번호, ...]}'
    )
    result = ollama_json(host, model, prompt, num_predict=600)
    theme = str(result.get("theme", "")).strip() or "(unspecified)"
    raw_indices = result.get("selected", [])
    indices: list[int] = []
    for value in raw_indices:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(sample) and index not in indices:
            indices.append(index)
    return theme, indices


# All structured workload axes are derived deterministically from the persona's
# OWN text salience + its valence — they are NOT guessed by the LLM. A persona is
# synthesized from forum text and has no poll data, so the only trustworthy
# signal is how much it talks about each axis (text_{axis}_tfidf). The LLM-scored
# variant was discarded: its values feed straight into the recommender (cosine of
# the full 16-dim persona vector in 'similarity' mode), so an arbitrary model
# guess would directly skew rankings. This unipolar, valence-driven derivation
# generalizes the old teamwork-aversion rule to every axis.
def structured_from_salience(
    text_features: dict[str, float], valence: str
) -> tuple[dict[str, float], float]:
    """Fill the 10 structured features from per-axis text salience.

    For each complementary pair we scale the axis's salience relative to the
    persona's strongest topic (peak) to get an intensity in [0, 1], then place it
    on the pole the persona's valence implies:
      avoid -> friendly pole (low / light / generous) — the easy class it seeks
      seek  -> burden  pole (high / heavy / strict)   — the substantial class it wants
    The opposite pole stays 0 (positive-unlabeled: never assert a preference we
    have no evidence for). Returns the features plus the teamwork intensity, kept
    in the meta as a sanity readout."""
    peak = max((float(value) for value in text_features.values()), default=0.0)
    features: dict[str, float] = {}
    for friendly_pole, burden_pole in STRUCTURED_PAIRS:
        axis = friendly_pole.split("_")[0]
        salience = float(text_features.get(f"text_{axis}_tfidf", 0.0))
        intensity = round(salience / peak, 6) if peak > 0.0 else 0.0
        if valence == "seek":
            features[friendly_pole] = 0.0
            features[burden_pole] = intensity
        else:
            features[friendly_pole] = intensity
            features[burden_pole] = 0.0
    teamwork_salience = float(text_features.get("text_teamwork_tfidf", 0.0))
    teamwork_intensity = round(teamwork_salience / peak, 6) if peak > 0.0 else 0.0
    return features, teamwork_intensity


# --------------------------------------------------------------------------- #
# Lecture nodes (restricted to the 16 persona columns)
# --------------------------------------------------------------------------- #
# Columns from a lecture-metadata file worth surfacing in the Top-K output,
# in display order. Any that are absent are simply skipped.
META_PREFERRED_COLUMNS = ["name", "professor", "code", "type", "credit", "lectureRate"]
META_ID_CANDIDATES = ["lectureId", "lecture_id", "id"]


def load_lecture_meta(path: Path) -> pd.DataFrame:
    """Load a lecture_id -> metadata table (csv or json) for merging course
    name / professor etc. into the recommendations. The id column is matched
    against the recommender's ``lecture_id``; duplicates keep the first row."""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            records = [{"lecture_id": key, **value} for key, value in raw.items()]
            meta = pd.DataFrame(records).astype(str)
        else:
            meta = pd.DataFrame(raw).astype(str)
    else:
        meta = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    id_column = next((column for column in META_ID_CANDIDATES if column in meta.columns), None)
    if id_column is None:
        raise ValueError(f"{path}: no lecture id column found (looked for {META_ID_CANDIDATES}).")

    keep = [column for column in META_PREFERRED_COLUMNS if column in meta.columns]
    if not keep:
        keep = [column for column in meta.columns if column != id_column]

    meta = meta[[id_column, *keep]].copy()
    meta = meta.rename(columns={id_column: "lecture_id"})
    meta["lecture_id"] = meta["lecture_id"].astype(str)
    meta = meta.dropna(subset=["lecture_id"]).drop_duplicates(subset=["lecture_id"], keep="first")
    return meta


def enrich_topk(topk: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Attach human-readable course info to the Top-K table: lecture metadata
    (name/professor/...) up front, plus the available keyword + sample-review
    columns at the end."""
    topk = topk.copy()
    topk["lecture_id"] = topk["lecture_id"].astype(str)

    info_columns: list[str] = []
    if args.lecture_meta and args.lecture_meta.exists():
        meta = load_lecture_meta(args.lecture_meta)
        info_columns = [column for column in meta.columns if column != "lecture_id"]
        topk = topk.merge(meta, on="lecture_id", how="left")

    # Move metadata columns directly after lecture_id for readability.
    if info_columns:
        ordered = ["rank", "lecture_id", *info_columns]
        rest = [column for column in topk.columns if column not in ordered]
        topk = topk[[column for column in ordered if column in topk.columns] + rest]

    keywords = load_keywords(args.keywords)
    examples = load_review_examples(args.articles)
    topk = topk.merge(keywords, on="lecture_id", how="left")
    topk = topk.merge(examples, on="lecture_id", how="left")
    return topk


def load_nodes_16(path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(path)
    required = {"lecture_id", "article_count", "rating_count", "rating_average", TARGET_COLUMN, *PERSONA_FEATURE_COLUMNS}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    nodes = nodes.dropna(subset=[TARGET_COLUMN, *PERSONA_FEATURE_COLUMNS]).copy()
    nodes["lecture_id"] = nodes["lecture_id"].astype(str)
    nodes[TARGET_COLUMN] = nodes[TARGET_COLUMN].astype(float)
    for column in PERSONA_FEATURE_COLUMNS:
        nodes[column] = nodes[column].astype(float)
    return nodes.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def category_biased_sample(
    posts: list[dict[str, Any]],
    target_category: str | None,
    sample_size: int,
    rng: random.Random,
    valence: str = "avoid",
) -> list[dict[str, Any]]:
    """Build the per-persona sample. With a target category, fill mostly from
    posts that mention it (so different personas explore different axes), then
    top up with random posts for context. Without one, sample uniformly.

    valence="seek" additionally restricts to praise/worth-it posts so the
    cluster reflects a challenge/quality-seeking student; falls back to the full
    pool if too few positive posts exist for the target axis."""
    sample_size = min(sample_size, len(posts))
    pool = posts
    if valence == "seek":
        positive = [post for post in posts if post["positive"]]
        if len(positive) >= max(sample_size // 2, 10):
            pool = positive
    if not target_category:
        return rng.sample(pool, min(sample_size, len(pool)))

    matching = [post for post in pool if target_category in post["categories"]]
    others = [post for post in pool if target_category not in post["categories"]]
    rng.shuffle(matching)
    rng.shuffle(others)
    # Keep ~80% on-axis so the dominant theme lands on the target category,
    # while leaving room for the model to find a specific angle within it.
    target_n = min(len(matching), max(int(sample_size * 0.8), sample_size - len(others)))
    chosen = matching[:target_n] + others[: sample_size - target_n]
    rng.shuffle(chosen)
    return chosen


def aggregate_similar_lectures(
    nodes: pd.DataFrame,
    seed_weights: dict[str, float],
    sim_columns: list[str],
    n_aggregate: int,
    rng: random.Random,
) -> tuple[dict[str, float], list[str], list[float]]:
    """Find the lectures most similar to the tendency seed and mean-pool their
    16-dim vectors into one persona vector.

    Similarity is cosine over the z-scored lecture matrix (the same
    standardization the recommender uses), restricted to ``sim_columns`` — all
    16 features in 'preference' seed mode, the 6 text features in 'text' mode.
    The top ``n_aggregate`` lectures' PERSONA_FEATURE_COLUMNS values are
    averaged, so the persona sits at the centroid of real, already-analyzed
    lectures and is treated as one student's preference vector."""
    x, _, _ = standardize(nodes[sim_columns].to_numpy(dtype=float))
    try:
        seed_vector = build_preference_vector(seed_weights, sim_columns)
        similarities = cosine_to_vector(x, seed_vector)
        order = np.argsort(-similarities)
    except ValueError:
        # Empty seed (no text signal at all): fall back to a random subset so a
        # persona is still produced.
        similarities = np.zeros(len(nodes), dtype=float)
        order = np.array(rng.sample(range(len(nodes)), len(nodes)))

    n_aggregate = max(1, min(n_aggregate, len(nodes)))
    top_index = order[:n_aggregate]
    chosen = nodes.iloc[top_index]

    weights = {
        column: round(float(chosen[column].mean()), 6) for column in PERSONA_FEATURE_COLUMNS
    }
    aggregated_ids = [str(value) for value in chosen["lecture_id"].tolist()]
    aggregated_sims = [round(float(similarities[i]), 6) for i in top_index]
    return weights, aggregated_ids, aggregated_sims


def build_one_persona(
    name: str,
    posts: list[dict[str, Any]],
    idf: dict[str, float],
    category_max: dict[str, float],
    nodes: pd.DataFrame,
    target_category: str | None,
    valence: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any]]:
    sample = category_biased_sample(posts, target_category, args.sample_size, rng, valence)
    sample_size = len(sample)

    theme, indices = select_theme_posts(
        args.host, args.model, sample, args.select_min, args.select_max, target_category
    )
    if len(indices) < args.select_min:
        # Fallback: model gave too few usable indices -> top up randomly so the
        # persona still gets a coherent-enough cluster size.
        pool = [i for i in range(len(sample)) if i not in indices]
        rng.shuffle(pool)
        indices = (indices + pool)[: args.select_min]
    indices = indices[: args.select_max]
    selected_posts = [sample[i] for i in indices]

    # 1) Tendency seed from the selected posts + comments.
    text_features = persona_text_features(selected_posts, idf, category_max)
    structured_features, teamwork_aversion = structured_from_salience(text_features, valence)
    seed = {**structured_features, **text_features}
    if args.seed_mode == "text":
        # Only the 6 topic-salience features steer the lecture search.
        sim_columns = TEXT_FEATURES
        seed_weights = {column: seed[column] for column in TEXT_FEATURES}
    else:
        # Full 16-dim preference (salience + valence-placed structured poles).
        sim_columns = PERSONA_FEATURE_COLUMNS
        seed_weights = {column: seed[column] for column in PERSONA_FEATURE_COLUMNS}

    # 2) Retrieve the lectures most similar to the seed and average a random
    # 10-20 of them: the persona is the centroid of real analyzed lectures.
    n_aggregate = rng.randint(args.agg_min, args.agg_max)
    weights, aggregated_ids, aggregated_sims = aggregate_similar_lectures(
        nodes, seed_weights, sim_columns, n_aggregate, rng
    )

    meta = {
        "persona": name,
        "theme": theme,
        "source": args.source,
        "target_category": target_category,
        "valence": valence,
        "model": args.model,
        "seed_mode": args.seed_mode,
        "sample_size": sample_size,
        "selected_count": len(selected_posts),
        "teamwork_aversion": teamwork_aversion,
        "n_aggregated_lectures": n_aggregate,
        "aggregated_lecture_ids": aggregated_ids,
        "aggregated_similarities": aggregated_sims,
        "seed_weights": {
            column: round(float(seed.get(column, 0.0)), 6) for column in PERSONA_FEATURE_COLUMNS
        },
        "selected_post_ids": [post["id"] for post in selected_posts],
        "selected_snippets": [
            {
                "post": post["snippet"][:120],
                "comments": [comment[:120] for comment in post["comments"]],
            }
            for post in selected_posts
        ],
        "weights": weights,
    }
    return weights, meta


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.source == "reviews":
        posts = load_reviews(args.articles)
    else:
        posts = load_about_class(args.about_dir)
    if len(posts) < args.select_min:
        raise ValueError(f"Not enough usable posts from source '{args.source}': {len(posts)}")
    print(f"[info] loaded {len(posts)} posts (source: {args.source})")

    print("[info] building lecture-corpus TF-IDF scale ...")
    idf, category_max = build_lecture_text_scale(args.articles)

    # Lecture nodes are always needed now: the persona vector is the centroid of
    # the lectures most similar to its tendency seed (not only for Top-K).
    nodes = load_nodes_16(args.nodes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    # Rotate the target axis across personas so a corpus skewed toward
    # assignment/exam complaints still yields varied personas.
    rotation = resolve_target_categories(args)

    for number in range(1, args.count + 1):
        name = f"persona_{number:03d}"
        target_category = rotation[(number - 1) % len(rotation)] if rotation else None
        # valence: mix alternates avoider/seeker so the set spans both directions.
        if args.valence == "mix":
            valence = "seek" if number % 2 == 0 else "avoid"
        else:
            valence = args.valence
        focus = f" [focus: {target_category or '-'} / {valence}]"
        print(f"\n[{number}/{args.count}] {name}{focus}: discovering theme via {args.model} ...")
        weights, meta = build_one_persona(name, posts, idf, category_max, nodes, target_category, valence, rng, args)
        print(f"    theme: {meta['theme']}")
        print(f"    selected {meta['selected_count']} posts")

        weights_path = args.out_dir / f"{name}.json"
        meta_path = args.out_dir / f"{name}_meta.json"
        weights_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    [OK] weights -> {weights_path}")

        row = {
            "persona": name,
            "target_category": target_category or "",
            "valence": valence,
            "theme": meta["theme"],
            "selected_count": meta["selected_count"],
        }

        if args.run_topk:
            recommendations = make_recommendations(
                nodes=nodes,
                feature_columns=PERSONA_FEATURE_COLUMNS,
                preference_weights=weights,
                similarity_weight=args.similarity_weight,
                quality_weight=args.quality_weight,
                alpha=args.alpha,
                score_mode=args.score_mode,
                valence=valence,
                ubiquity_penalty=args.ubiquity_penalty,
            )
            topk = select_topk(recommendations, args.top_k, feature_columns=PERSONA_FEATURE_COLUMNS)
            topk = enrich_topk(topk, args)
            topk_path = args.out_dir / f"{name}_topk_{args.top_k}.csv"
            topk.to_csv(topk_path, index=False, encoding="utf-8-sig")
            row["top_lecture_ids"] = ", ".join(topk["lecture_id"].astype(str).head(args.top_k))
            if "name" in topk.columns:
                labels = [
                    f"{nm}({pf})" if isinstance(pf, str) and pf else str(nm)
                    for nm, pf in zip(topk.get("name", []), topk.get("professor", [""] * len(topk)))
                ]
                row["top_courses"] = " | ".join(label for label in labels[:5] if label and label != "nan")
            print(f"    [OK] top-{args.top_k} -> {topk_path}")
            print(f"    top lectures: {row['top_lecture_ids']}")
            if row.get("top_courses"):
                print(f"    top courses: {row['top_courses']}")

        index_rows.append(row)

    index_path = args.out_dir / "personas_index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] wrote persona index: {index_path}")


def resolve_target_categories(args: argparse.Namespace) -> list[str]:
    """Decide the per-persona target-axis rotation. [] = uniform sampling."""
    if args.no_diverse:
        return []
    if args.axis:
        return [args.axis]
    return list(CATEGORY_KEYWORDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build persona test vectors from about-class and (optionally) run Top-K.")
    parser.add_argument("--count", type=int, default=1, help="Number of personas to generate.")
    parser.add_argument(
        "--source",
        choices=["about-class", "reviews"],
        default="about-class",
        help="Where persona posts come from. about-class (default) = forum posts in --about-dir. "
        "reviews = lecture reviews in --articles (lecture_articles.csv), the SAME text the lecture "
        "TF-IDF features were built from.",
    )
    parser.add_argument("--about-dir", type=Path, default=Path("about-class"))
    parser.add_argument("--articles", type=Path, default=Path("data/normalized/lecture_articles.csv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument(
        "--lecture-meta",
        type=Path,
        default=Path("subjects.csv"),
        help="lecture_id -> course name/professor table (csv or json) merged into Top-K. Matched on lectureId/lecture_id/id.",
    )
    parser.add_argument("--keywords", type=Path, default=Path("data/model/lecture_top_keywords.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/personas"))
    parser.add_argument("--model", default="qwen3.5:27b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--sample-size", type=int, default=100, help="Posts randomly sampled per persona.")
    parser.add_argument("--select-min", type=int, default=10)
    parser.add_argument("--select-max", type=int, default=20)
    parser.add_argument(
        "--seed-mode",
        choices=["preference", "text"],
        default="preference",
        help="What the persona's tendency seed is, used to retrieve similar lectures. "
        "preference (default) = full 16-dim vector (posts/comments TF-IDF text-6 + "
        "valence-placed structured-10). text = the 6 topic-salience features only.",
    )
    parser.add_argument(
        "--agg-min",
        type=int,
        default=10,
        help="Min number of similar lectures averaged into the persona vector.",
    )
    parser.add_argument(
        "--agg-max",
        type=int,
        default=20,
        help="Max number of similar lectures averaged into the persona vector "
        "(the actual count is drawn uniformly from [agg-min, agg-max] per persona).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--axis",
        choices=sorted(CATEGORY_KEYWORDS),
        default=None,
        help="Force every persona to center on this axis (assignment/exam/teamwork/attendance/grading/teaching).",
    )
    parser.add_argument(
        "--no-diverse",
        action="store_true",
        help="Disable category-rotation sampling; sample posts uniformly (old behavior).",
    )
    parser.add_argument("--no-run-topk", dest="run_topk", action="store_false", help="Only emit persona weights, skip recommendation.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--score-mode",
        choices=["similarity", "objective", "objective_rel"],
        default="similarity",
        help="similarity (default) = cosine of the FULL 16-dim persona vector vs the "
        "z-scored lecture matrix; the deterministically-derived structured features are "
        "actually used, so the persona's workload preference drives the ranking. "
        "objective = per-axis text-salience weights x lecture poll-based friendliness "
        "(structured features unused). objective_rel = objective with per-axis z-scoring.",
    )
    parser.add_argument(
        "--valence",
        choices=["avoid", "seek", "mix"],
        default="avoid",
        help="avoid (default) = burden-averse personas -> low-burden classes. "
        "seek = challenge/quality-seeking personas (samples praise posts, scores the opposite "
        "pole) -> substantial classes. mix = alternate avoid/seek across personas for diversity.",
    )
    parser.add_argument(
        "--ubiquity-penalty",
        type=float,
        default=0.0,
        help="Idea C (objective modes): subtract this x a lecture's generic friendliness so "
        "'easy on everything' seminars stop topping every list. ~0.3-0.5 adds within-group "
        "diversity at a small fit cost. 0 = off.",
    )
    parser.add_argument("--similarity-weight", type=float, default=0.7)
    parser.add_argument("--quality-weight", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.set_defaults(run_topk=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
