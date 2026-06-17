from __future__ import annotations

import os
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from config import (
    ARTICLE_LIST_API_URL,
    COOKIE_ENV_NAME,
    DEFAULT_HEADERS,
    DEFAULT_LIMIT,
    DEFAULT_OFFSET,
    DEFAULT_SORT,
    LECTURE_API_URL,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    SUCCESS_STATUS,
)


class LectureApiError(RuntimeError):
    """Raised when the lecture API request or response handling fails."""


@dataclass
class LectureApiClient:
    """Small client for requesting lecture review data from the internal API."""

    lecture_api_url: str = LECTURE_API_URL
    article_list_api_url: str = ARTICLE_LIST_API_URL
    headers: dict[str, str] = field(default_factory=lambda: DEFAULT_HEADERS.copy())
    timeout: int = REQUEST_TIMEOUT_SECONDS
    limit: int = DEFAULT_LIMIT
    offset: int = DEFAULT_OFFSET
    sort: str = DEFAULT_SORT
    request_delay_seconds: float = REQUEST_DELAY_SECONDS
    cookie: str | None = None
    last_request_at: float = 0.0

    # 요청 헤더를 구성하고 환경변수의 로그인 쿠키를 포함합니다.
    def build_headers(self) -> dict[str, str]:
        """Build request headers, including the login cookie when available."""
        headers = self.headers.copy()
        cookie = self.cookie or os.getenv(COOKIE_ENV_NAME)

        if cookie:
            headers["Cookie"] = cookie

        return headers

    # 직전 요청과 현재 요청 사이에 최소 대기 시간을 보장합니다.
    def wait_for_rate_limit(self) -> None:
        """Wait until the configured request interval has passed."""
        if self.request_delay_seconds <= 0:
            return

        elapsed = time.monotonic() - self.last_request_at
        remaining = self.request_delay_seconds - elapsed

        if remaining > 0:
            time.sleep(remaining)

    # 강의 상세 API에 보낼 form payload를 생성합니다.
    def build_lecture_payload(
        self,
        lecture_id: int,
        limit: int | None = None,
        offset: int | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Build the POST request body for one lecture id.

        The current form payload is based on the browser Network tab:
        id, limit, offset, and sort.
        """
        return {
            "id": lecture_id,
            "limit": self.limit if limit is None else limit,
            "offset": self.offset if offset is None else offset,
            "sort": self.sort if sort is None else sort,
        }

    # 강의평가 글 목록 API에 보낼 form payload를 생성합니다.
    def build_article_list_payload(
        self,
        lecture_id: int,
        limit: int | None = None,
        offset: int | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Build the POST body for lecture review articles.

        This endpoint expects lectureId instead of id.
        """
        return {
            "lectureId": lecture_id,
            "limit": self.limit if limit is None else limit,
            "offset": self.offset if offset is None else offset,
            "sort": self.sort if sort is None else sort,
        }

    # form-encoded POST 요청을 보내고 JSON 응답 상태를 검증합니다.
    def post_form(
        self,
        url: str,
        payload: dict[str, Any],
        lecture_id: int,
    ) -> dict[str, Any]:
        """Send a form-encoded POST request and validate the JSON response."""
        self.wait_for_rate_limit()

        try:
            encoded_payload = urllib.parse.urlencode(payload).encode("utf-8")
            request = urllib.request.Request(
                url=url,
                data=encoded_payload,
                headers=self.build_headers(),
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")

            self.last_request_at = time.monotonic()
        except (urllib.error.URLError, TimeoutError) as exc:
            self.last_request_at = time.monotonic()
            raise LectureApiError(
                f"Failed to request lecture_id={lecture_id}: {exc}"
            ) from exc

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LectureApiError(
                f"API response for lecture_id={lecture_id} is not valid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise LectureApiError(
                f"Expected JSON object for lecture_id={lecture_id}, "
                f"but got {type(data).__name__}."
            )

        if data.get("status") != SUCCESS_STATUS:
            raise LectureApiError(
                f"API returned status={data.get('status')!r} for lecture_id={lecture_id}."
            )

        return data

    # 강의 상세 데이터를 요청합니다.
    def fetch_lecture(
        self,
        lecture_id: int,
        limit: int | None = None,
        offset: int | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Request one lecture's JSON data from the internal API."""
        payload = self.build_lecture_payload(
            lecture_id=lecture_id,
            limit=limit,
            offset=offset,
            sort=sort,
        )
        return self.post_form(self.lecture_api_url, payload, lecture_id)

    # 강의평가 글 목록 데이터를 요청합니다.
    def fetch_article_list(
        self,
        lecture_id: int,
        limit: int | None = None,
        offset: int | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Request review article list JSON for one lecture."""
        payload = self.build_article_list_payload(
            lecture_id=lecture_id,
            limit=limit,
            offset=offset,
            sort=sort,
        )
        return self.post_form(self.article_list_api_url, payload, lecture_id)

    # 하나의 lecture id에 대해 상세 데이터와 글 목록 데이터를 함께 요청합니다.
    def fetch_lecture_bundle(self, lecture_id: int) -> dict[str, dict[str, Any]]:
        """Request both detail and article-list JSON for one lecture."""
        return {
            "detail": self.fetch_lecture(lecture_id),
            "articles": self.fetch_article_list(lecture_id),
        }

    # 여러 lecture id에 대해 강의 상세 데이터를 요청합니다.
    def fetch_lectures(
        self,
        lecture_ids: list[int],
        limit: int | None = None,
        offset: int | None = None,
        sort: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Request multiple lectures and return data indexed by lecture id."""
        results: dict[int, dict[str, Any]] = {}

        for lecture_id in lecture_ids:
            results[lecture_id] = self.fetch_lecture(
                lecture_id=lecture_id,
                limit=limit,
                offset=offset,
                sort=sort,
            )

        return results
