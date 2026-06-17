from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_FEATURE_COLUMNS = [
    "rating_average_norm",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in [None, ""]:
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


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
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def article_counts(article_rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in article_rows:
        lecture_id = row["lecture_id"]
        counts[lecture_id] = counts.get(lecture_id, 0) + 1
    return counts


def build_node(row: dict[str, str], article_count: int) -> dict[str, Any]:
    rating_average = as_float(row.get("rating_average"))

    assignment_none = as_float(row.get("assignment_없음_ratio"))
    assignment_normal = as_float(row.get("assignment_보통_ratio"))
    assignment_many = as_float(row.get("assignment_많음_ratio"))

    teamwork_none = as_float(row.get("teamwork_없음_ratio"))
    teamwork_normal = as_float(row.get("teamwork_보통_ratio"))
    teamwork_many = as_float(row.get("teamwork_많음_ratio"))

    grading_generous = as_float(row.get("grading_너그러움_ratio"))
    grading_normal = as_float(row.get("grading_보통_ratio"))
    grading_strict = as_float(row.get("grading_깐깐함_ratio"))

    attendance_none = as_float(row.get("attendance_반영안함_ratio"))
    attendance_electronic = as_float(row.get("attendance_전자출결_ratio"))
    attendance_called = as_float(row.get("attendance_직접호명_ratio"))
    attendance_assigned = as_float(row.get("attendance_지정좌석_ratio"))
    attendance_mixed = as_float(row.get("attendance_복합적_ratio"))

    exam_none = as_float(row.get("exam_없음_ratio"))
    exam_once = as_float(row.get("exam_once_ratio"))
    exam_twice = as_float(row.get("exam_twice_ratio"))
    exam_three = as_float(row.get("exam_three_times_ratio"))
    exam_four_plus = as_float(row.get("exam_four_plus_ratio"))

    node = {
        "lecture_id": row["lecture_id"],
        "article_count": article_count,
        "rating_count": int(as_float(row.get("rating_count"))),
        "rating_average": rating_average,
        "rating_average_norm": round(rating_average / 5.0, 6) if rating_average else 0.0,
        "assignment_low_score": round(assignment_none + 0.5 * assignment_normal, 6),
        "assignment_high_score": round(assignment_many + 0.5 * assignment_normal, 6),
        "teamwork_low_score": round(teamwork_none + 0.5 * teamwork_normal, 6),
        "teamwork_high_score": round(teamwork_many + 0.5 * teamwork_normal, 6),
        "grading_generous_score": round(grading_generous + 0.5 * grading_normal, 6),
        "grading_strict_score": round(grading_strict + 0.5 * grading_normal, 6),
        "attendance_light_score": round(attendance_none + 0.5 * attendance_electronic, 6),
        "attendance_strict_score": round(attendance_called + attendance_assigned + 0.5 * attendance_mixed, 6),
        "exam_light_score": round(exam_none + 0.75 * exam_once + 0.4 * exam_twice, 6),
        "exam_heavy_score": round(0.4 * exam_twice + 0.75 * exam_three + exam_four_plus, 6),
    }

    return node


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lecture node features for recommendation experiments.")
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/normalized"))
    parser.add_argument("--out", type=Path, default=Path("data/model/lecture_nodes.csv"))
    parser.add_argument("--min-articles", type=int, default=5)
    parser.add_argument("--min-rating-count", type=int, default=5)
    args = parser.parse_args()

    articles = read_csv(args.normalized_dir / "lecture_articles.csv")
    details = read_csv(args.normalized_dir / "lecture_details.csv")
    counts = article_counts(articles)

    nodes: list[dict[str, Any]] = []
    skipped_few_articles = 0
    skipped_few_ratings = 0

    for row in details:
        lecture_id = row["lecture_id"]
        article_count = counts.get(lecture_id, 0)
        rating_count = int(as_float(row.get("rating_count")))

        if article_count < args.min_articles:
            skipped_few_articles += 1
            continue
        if rating_count < args.min_rating_count:
            skipped_few_ratings += 1
            continue

        nodes.append(build_node(row, article_count))

    write_csv(args.out, nodes)

    print(f"[OK] lecture nodes: {len(nodes)}")
    print(f"[OK] skipped by article_count < {args.min_articles}: {skipped_few_articles}")
    print(f"[OK] skipped by rating_count < {args.min_rating_count}: {skipped_few_ratings}")
    print(f"[OK] feature columns: {', '.join(DEFAULT_FEATURE_COLUMNS)}")
    print(f"[OK] output: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
