from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")

STOPWORDS = {
    "그리고",
    "그냥",
    "근데",
    "너무",
    "정말",
    "진짜",
    "수업",
    "강의",
    "교수",
    "교수님",
    "합니다",
    "했습니다",
    "있습니다",
    "같습니다",
    "입니다",
    "합니다",
    "해서",
    "하면",
    "제가",
    "저는",
    "것",
    "거",
    "때",
    "좀",
    "더",
    "잘",
    "많이",
}

CATEGORY_KEYWORDS = {
    "assignment": {
        "과제",
        "레포트",
        "리포트",
        "숙제",
        "제출",
        "퀴즈",
        "보고서",
        "문제풀이",
    },
    "exam": {
        "시험",
        "중간",
        "기말",
        "고사",
        "오픈북",
        "족보",
        "문제",
        "객관식",
        "주관식",
        "논술",
        "암기",
    },
    "teamwork": {
        "팀플",
        "조별",
        "조모임",
        "발표",
        "프로젝트",
        "조원",
        "협업",
    },
    "attendance": {
        "출석",
        "출결",
        "결석",
        "지각",
        "호명",
        "전자출결",
        "지정좌석",
    },
    "grading": {
        "학점",
        "성적",
        "에이쁠",
        "에이플",
        "A+",
        "비쁠",
        "후함",
        "너그러움",
        "깐깐",
        "절평",
        "상평",
    },
    "teaching": {
        "설명",
        "강의력",
        "전달",
        "이해",
        "재미",
        "지루",
        "친절",
        "자료",
        "피피티",
        "ppt",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(str(text).lower()):
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def sentiment_from_rate(rate: str) -> int:
    try:
        value = int(float(rate))
    except (TypeError, ValueError):
        return 0
    if value >= 4:
        return 1
    if value <= 2:
        return -1
    return 0


def compute_idf(tokenized_docs: list[list[str]]) -> dict[str, float]:
    document_count = len(tokenized_docs)
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))
    return {
        token: math.log((1 + document_count) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }


def category_scores(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    scores = {category: 0.0 for category in CATEGORY_KEYWORDS}
    counts = Counter(tokens)
    for token, count in counts.items():
        for category, keywords in CATEGORY_KEYWORDS.items():
            if token in {keyword.lower() for keyword in keywords}:
                scores[category] += count * idf.get(token, 1.0)
    return scores


def normalize_scores(rows: list[dict[str, Any]], columns: list[str]) -> None:
    max_by_column: dict[str, float] = {}
    for column in columns:
        max_by_column[column] = max((abs(float(row.get(column, 0.0))) for row in rows), default=0.0)

    for row in rows:
        for column in columns:
            max_value = max_by_column[column]
            value = float(row.get(column, 0.0))
            row[column] = round(value / max_value, 6) if max_value else 0.0


def build_text_features(article_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenized_docs = [tokenize(row.get("text", "")) for row in article_rows]
    idf = compute_idf(tokenized_docs)

    lecture_feature_rows: dict[str, dict[str, Any]] = {}
    lecture_top_terms: dict[str, Counter[str]] = defaultdict(Counter)

    for row, tokens in zip(article_rows, tokenized_docs):
        lecture_id = row["lecture_id"]
        sentiment = sentiment_from_rate(row.get("rate", ""))
        scores = category_scores(tokens, idf)
        target = lecture_feature_rows.setdefault(
            lecture_id,
            {
                "lecture_id": lecture_id,
                "text_review_count": 0,
                "text_positive_count": 0,
                "text_negative_count": 0,
                "text_neutral_count": 0,
            },
        )
        target["text_review_count"] += 1
        if sentiment > 0:
            target["text_positive_count"] += 1
        elif sentiment < 0:
            target["text_negative_count"] += 1
        else:
            target["text_neutral_count"] += 1

        for category, score in scores.items():
            target[f"text_{category}_tfidf"] = target.get(f"text_{category}_tfidf", 0.0) + score
            if sentiment > 0:
                target[f"text_{category}_positive_tfidf"] = target.get(f"text_{category}_positive_tfidf", 0.0) + score
            elif sentiment < 0:
                target[f"text_{category}_negative_tfidf"] = target.get(f"text_{category}_negative_tfidf", 0.0) + score

        weighted_terms = Counter({token: idf.get(token, 1.0) for token in tokens})
        lecture_top_terms[lecture_id].update(weighted_terms)

    rows = list(lecture_feature_rows.values())
    feature_columns = []
    for category in CATEGORY_KEYWORDS:
        feature_columns.extend(
            [
                f"text_{category}_tfidf",
                f"text_{category}_positive_tfidf",
                f"text_{category}_negative_tfidf",
            ]
        )

    for row in rows:
        review_count = max(int(row["text_review_count"]), 1)
        row["text_positive_ratio"] = round(row["text_positive_count"] / review_count, 6)
        row["text_negative_ratio"] = round(row["text_negative_count"] / review_count, 6)
        row["text_neutral_ratio"] = round(row["text_neutral_count"] / review_count, 6)
        for column in feature_columns:
            row[column] = float(row.get(column, 0.0)) / review_count

    normalize_scores(rows, feature_columns)

    keyword_rows = []
    for lecture_id, counter in lecture_top_terms.items():
        top_terms = [term for term, _ in counter.most_common(20)]
        keyword_rows.append({"lecture_id": lecture_id, "top_tfidf_terms": ", ".join(top_terms)})

    return rows, keyword_rows


def merge_features(nodes: list[dict[str, str]], text_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_lecture_id = {row["lecture_id"]: row for row in text_rows}
    merged = []
    text_columns = sorted({key for row in text_rows for key in row if key != "lecture_id"})

    for node in nodes:
        merged_row: dict[str, Any] = dict(node)
        text_row = by_lecture_id.get(node["lecture_id"], {})
        for column in text_columns:
            merged_row[column] = text_row.get(column, 0)
        merged.append(merged_row)

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TF-IDF and sentiment text features from lecture review articles.")
    parser.add_argument("--articles", type=Path, default=Path("data/normalized/lecture_articles.csv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes.csv"))
    parser.add_argument("--out-features", type=Path, default=Path("data/model/lecture_text_features.csv"))
    parser.add_argument("--out-nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--out-keywords", type=Path, default=Path("data/model/lecture_top_keywords.csv"))
    args = parser.parse_args()

    articles = read_csv(args.articles)
    nodes = read_csv(args.nodes)
    text_features, keyword_rows = build_text_features(articles)
    merged_nodes = merge_features(nodes, text_features)

    write_csv(args.out_features, text_features)
    write_csv(args.out_nodes, merged_nodes)
    write_csv(args.out_keywords, keyword_rows)

    print(f"[OK] article rows used: {len(articles)}")
    print(f"[OK] lectures with text features: {len(text_features)}")
    print(f"[OK] merged lecture nodes: {len(merged_nodes)}")
    print(f"[OK] text features: {args.out_features.resolve()}")
    print(f"[OK] nodes with text: {args.out_nodes.resolve()}")
    print(f"[OK] top keywords: {args.out_keywords.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
