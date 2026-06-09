from __future__ import annotations

from pathlib import Path


# [수정 포인트] 프로젝트 루트 경로입니다. 일반적으로 수정하지 않아도 됩니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env vars."""
    if not path.exists():
        return

    import os

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

# [수정 포인트] 수집할 lecture id 목록 파일 경로입니다.
LECTURE_IDS_PATH = PROJECT_ROOT / "data" / "lecture_ids.json"

# [수정 포인트] API 원본 응답 JSON을 저장할 디렉터리입니다.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# [수정 포인트] 강의 상세 데이터를 요청하는 내부 API URL입니다.
LECTURE_API_URL = "https://api.everytime.kr/find/lecture"

# [수정 포인트] 강의평가 글 목록을 요청하는 내부 API URL입니다.
ARTICLE_LIST_API_URL = "https://api.everytime.kr/find/lecture/article/list"

# [수정 포인트] 로그인 세션 쿠키를 읽어올 환경변수 이름입니다.
COOKIE_ENV_NAME = "EVERYTIME_COOKIE"

# [수정 포인트] API 요청에 사용할 기본 HTTP headers입니다.
DEFAULT_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "User-Agent": "Mozilla/5.0",
}

# [수정 포인트] 한 번에 요청할 데이터 개수입니다.
DEFAULT_LIMIT = 20

# [수정 포인트] 페이지네이션 시작 위치입니다.
DEFAULT_OFFSET = 40

# [수정 포인트] API 응답 정렬 기준입니다.
DEFAULT_SORT = "id"

# [수정 포인트] 반복 요청 차단을 피하기 위한 요청 간 최소 대기 시간입니다.
REQUEST_DELAY_SECONDS = 1.5

# [수정 포인트] API가 정상 처리되었을 때 반환하는 status 값입니다.
SUCCESS_STATUS = "success"

# [수정 포인트] HTTP 요청 응답을 기다리는 최대 시간입니다.
REQUEST_TIMEOUT_SECONDS = 10
