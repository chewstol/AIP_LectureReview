from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


LECTURE_FILE_RE = re.compile(r"lecture_(\d+)_(articles|detail)\.json$")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] skipped {path}: {exc}")
        return None

    if not isinstance(data, dict):
        print(f"[WARN] skipped {path}: top-level JSON is not an object")
        return None

    return data


def lecture_id_from_path(path: Path) -> int | None:
    match = LECTURE_FILE_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def count_lookup(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        text = clean_text(item.get("text"))
        if text:
            counts[text] = int(item.get("count") or 0)
    return counts


def ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 6)


def append_review_distribution(row: dict[str, Any], detail: dict[str, Any]) -> None:
    review = detail.get("review") or {}
    rate = review.get("rate") or {}
    row["rating_average"] = rate.get("average")
    row["rating_count"] = rate.get("count")

    for item in rate.get("items") or []:
        value = item.get("value")
        if value in [1, 2, 3, 4, 5]:
            row[f"rating_{value}_count"] = int(item.get("count") or 0)

    subjective = review.get("subjectiveDetails") or []
    objective = review.get("objectiveDetails") or []

    name_to_prefix = {
        "과제": ("assignment", ["없음", "보통", "많음"]),
        "조모임": ("teamwork", ["없음", "보통", "많음"]),
        "성적": ("grading", ["너그러움", "보통", "깐깐함"]),
        "출결": ("attendance", ["반영안함", "전자출결", "지정좌석", "직접호명", "복합적"]),
        "시험": ("exam", ["없음", "한 번", "두 번", "세 번", "네 번 이상"]),
    }

    for group in subjective + objective:
        name = clean_text(group.get("name"))
        if name not in name_to_prefix:
            continue

        prefix, labels = name_to_prefix[name]
        total = int(group.get("count") or 0)
        row[f"{prefix}_response_count"] = total

        counts = count_lookup(group.get("items") or [])
        for label in labels:
            safe_label = (
                label.replace(" ", "_")
                .replace("한_번", "once")
                .replace("두_번", "twice")
                .replace("세_번", "three_times")
                .replace("네_번_이상", "four_plus")
            )
            key = f"{prefix}_{safe_label}"
            count = counts.get(label, 0)
            row[f"{key}_count"] = count
            row[f"{key}_ratio"] = ratio(count, total)


def normalize_exam_rows(lecture_id: int, detail: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exam = detail.get("exam") or {}
    for question_type in exam.get("questionTypes") or []:
        nth = clean_text(question_type.get("nth"))
        total = int(question_type.get("count") or 0)
        for item in question_type.get("items") or []:
            rows.append(
                {
                    "lecture_id": lecture_id,
                    "exam_nth": nth,
                    "exam_response_count": total,
                    "question_type": clean_text(item.get("text")),
                    "question_type_count": int(item.get("count") or 0),
                    "question_type_ratio": ratio(int(item.get("count") or 0), total),
                }
            )
    return rows


def normalize_book_rows(lecture_id: int, detail: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for book in detail.get("books") or []:
        rows.append(
            {
                "lecture_id": lecture_id,
                "book_isbn": clean_text(book.get("bookIsbn")),
                "book_title": clean_text(book.get("bookTitle")),
                "book_author": clean_text(book.get("bookAuthor")),
                "book_publisher": clean_text(book.get("bookPublisher")),
                "book_price": book.get("bookPrice"),
                "used_price": book.get("price"),
                "cover_image": clean_text(book.get("coverImage")),
            }
        )
    return rows


def article_rows_from_file(path: Path, lecture_id: int) -> list[dict[str, Any]]:
    data = load_json(path)
    if not data:
        return []

    articles = ((data.get("result") or {}).get("articles") or [])
    rows: list[dict[str, Any]] = []

    for article in articles:
        if not isinstance(article, dict):
            continue
        text = clean_text(article.get("text"))
        rows.append(
            {
                "lecture_id": lecture_id,
                "article_id": article.get("id"),
                "year": article.get("year"),
                "semester": clean_text(article.get("semester")),
                "rate": article.get("rate"),
                "posvote": article.get("posvote"),
                "text": text,
                "text_length": len(text),
                "text_hash": text_hash(text),
                "source": "articles",
                "source_file": path.name,
            }
        )

    return rows


def detail_outputs_from_file(path: Path, lecture_id: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    data = load_json(path)
    if not data:
        return None, [], []

    detail = data.get("result") or {}
    if not isinstance(detail, dict):
        return None, [], []

    row: dict[str, Any] = {
        "lecture_id": lecture_id,
        "exam_knowhow": clean_text((detail.get("exam") or {}).get("knowhow")),
    }
    append_review_distribution(row, detail)

    return row, normalize_exam_rows(lecture_id, detail), normalize_book_rows(lecture_id, detail)


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


def dedupe_articles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any, Any]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row["lecture_id"], row.get("article_id"), row.get("year"), row.get("text_hash"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Everytime raw lecture JSON files into CSV tables.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_import/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/normalized"))
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    if not raw_dir.exists():
        print(f"[ERROR] raw directory does not exist: {raw_dir}")
        return 1

    article_rows: list[dict[str, Any]] = []
    lecture_rows: list[dict[str, Any]] = []
    exam_rows: list[dict[str, Any]] = []
    book_rows: list[dict[str, Any]] = []

    for path in sorted(raw_dir.glob("lecture_*_articles.json")):
        lecture_id = lecture_id_from_path(path)
        if lecture_id is not None:
            article_rows.extend(article_rows_from_file(path, lecture_id))

    for path in sorted(raw_dir.glob("lecture_*_detail.json")):
        lecture_id = lecture_id_from_path(path)
        if lecture_id is None:
            continue
        lecture_row, lecture_exam_rows, lecture_book_rows = detail_outputs_from_file(path, lecture_id)
        if lecture_row:
            lecture_rows.append(lecture_row)
        exam_rows.extend(lecture_exam_rows)
        book_rows.extend(lecture_book_rows)

    article_rows = dedupe_articles(article_rows)

    write_csv(out_dir / "lecture_articles.csv", article_rows)
    write_csv(out_dir / "lecture_details.csv", lecture_rows)
    write_csv(out_dir / "exam_question_types.csv", exam_rows)
    write_csv(out_dir / "books.csv", book_rows)

    print(f"[OK] lecture_articles.csv: {len(article_rows)} rows")
    print(f"[OK] lecture_details.csv: {len(lecture_rows)} rows")
    print(f"[OK] exam_question_types.csv: {len(exam_rows)} rows")
    print(f"[OK] books.csv: {len(book_rows)} rows")
    print(f"[OK] output directory: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
