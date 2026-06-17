from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api_client import LectureApiClient, LectureApiError
from config import LECTURE_IDS_PATH, RAW_DATA_DIR


# lecture id 목록을 JSON 파일에서 읽어옵니다.
def load_lecture_ids(path: Path = LECTURE_IDS_PATH) -> list[int]:
    """Load lecture ids from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        lecture_ids = json.load(file)

    if not isinstance(lecture_ids, list):
        raise ValueError(f"Expected a list in {path}, but got {type(lecture_ids).__name__}.")

    return [int(lecture_id) for lecture_id in lecture_ids]


# API 응답 데이터를 보기 좋은 JSON 파일로 저장합니다.
def save_json(data: dict[str, Any], path: Path) -> None:
    """Save one API response as a pretty-printed JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# lecture id별 상세 데이터와 글 목록 데이터를 수집합니다.
def collect() -> None:
    """Collect detail and article-list JSON files for each lecture id."""
    lecture_ids = load_lecture_ids()
    client = LectureApiClient()

    for lecture_id in lecture_ids:
        try:
            bundle = client.fetch_lecture_bundle(lecture_id)
        except LectureApiError as exc:
            print(f"[FAIL] lecture_id={lecture_id}: {exc}")
            continue

        detail_path = RAW_DATA_DIR / f"lecture_{lecture_id}_detail.json"
        articles_path = RAW_DATA_DIR / f"lecture_{lecture_id}_articles.json"

        save_json(bundle["detail"], detail_path)
        save_json(bundle["articles"], articles_path)

        print(f"[OK] lecture_id={lecture_id} -> {detail_path}")
        print(f"[OK] lecture_id={lecture_id} -> {articles_path}")


if __name__ == "__main__":
    collect()
