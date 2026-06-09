# Lecture Review Data Collector

강의평가 사이트의 내부 API를 활용하여 강의평가 데이터를 구조화된 JSON 파일로 수집하는 Python 기반 데이터 수집 프로젝트입니다.

수집한 데이터는 이후 AI 기반 강의 추천 시스템, 그래프 기반 분석, GNN(Graph Neural Network) 학습 데이터로 활용하는 것을 목표로 합니다.

## 중요 지침

프로젝트 디렉터리 내의 모든 Python 파일에서 함수를 선언할 때는 함수 선언 한 줄 위에 함수의 역할을 간단하게 설명하는 주석을 추가합니다.

예시는 다음과 같습니다.

```python
# lecture id 목록을 JSON 파일에서 읽어옵니다.
def load_lecture_ids(path):
    ...
```

주석은 함수가 맡는 역할을 짧게 설명하는 데 집중합니다. 코드 내용을 그대로 반복하는 장황한 주석은 피하고, 이후 유지보수자가 파일을 훑을 때 흐름을 빠르게 이해할 수 있도록 작성합니다.

## 프로젝트 개요

`Lecture Review Data Collector`는 HTML 페이지를 직접 파싱하는 방식보다, 브라우저 개발자도구의 Network 탭에서 확인한 내부 API 요청을 재현하는 방식으로 데이터를 수집합니다.

현재 MVP에서는 `data/lecture_ids.json`에 저장된 lecture id 목록을 읽고, 각 id에 대해 다음 두 API를 호출합니다.

| 구분 | Request URL | payload id key | 저장 파일 |
| --- | --- | --- | --- |
| 강의 상세 데이터 | `https://api.everytime.kr/find/lecture` | `id` | `lecture_<id>_detail.json` |
| 강의평가 글 목록 | `https://api.everytime.kr/find/lecture/article/list` | `lectureId` | `lecture_<id>_articles.json` |

두 endpoint는 비슷해 보이지만 lecture id를 전달하는 key가 다릅니다. `find/lecture`는 `id`를 사용해야 하고, `article/list`는 `lectureId`를 사용해야 정상 응답을 받을 수 있습니다.

## 동작 원리

1. `data/lecture_ids.json`에서 수집할 lecture id 목록을 읽습니다.
2. lecture id마다 강의 상세 API에 POST 요청을 보냅니다.
3. 같은 lecture id로 강의평가 글 목록 API에도 POST 요청을 보냅니다.
4. API 응답의 `status` 값이 `success`인지 확인합니다.
5. 성공한 응답만 `data/raw/` 아래에 JSON 파일로 저장합니다.

현재 확인된 기본 payload는 다음과 같습니다.

강의 상세 데이터 요청:

```json
{
  "id": 1937322,
  "limit": 20,
  "offset": 40,
  "sort": "id"
}
```

강의평가 글 목록 요청:

```json
{
  "lectureId": 1937322,
  "limit": 20,
  "offset": 40,
  "sort": "id"
}
```

요청은 `application/x-www-form-urlencoded` 형식의 POST 요청으로 전송합니다.

## 현재 프로젝트 구조

```text
Lecture-Review-Data-Collector/
├── README.md
├── requirements.txt
├── data/
│   ├── lecture_ids.json
│   └── raw/
│       ├── lecture_488935_detail.json
│       ├── lecture_488935_articles.json
│       ├── lecture_1937322_detail.json
│       └── lecture_1937322_articles.json
└── src/
    ├── api_client.py
    ├── config.py
    └── collect.py
```

각 파일의 역할은 다음과 같습니다.

| 경로 | 설명 |
| --- | --- |
| `data/lecture_ids.json` | 수집 대상 lecture id 목록입니다. 현재 샘플 id는 `488935`, `1937322`입니다. |
| `data/raw/` | API에서 받은 원본 JSON 응답을 저장합니다. |
| `src/api_client.py` | 내부 API에 POST 요청을 보내고 응답을 검증합니다. |
| `src/config.py` | API URL, 요청 지연 시간, 기본 payload 값, 저장 경로 등 수정이 잦은 설정값을 관리합니다. |
| `src/collect.py` | lecture id 목록을 읽고, id별 detail/articles JSON 파일을 저장합니다. |
| `requirements.txt` | 실행에 필요한 Python 패키지 목록입니다. |

## 데이터 흐름

```text
data/lecture_ids.json
        ↓
lecture id 목록 로딩
        ↓
find/lecture POST 요청
        ↓
lecture_<id>_detail.json 저장
        ↓
find/lecture/article/list POST 요청
        ↓
lecture_<id>_articles.json 저장
```

하나의 lecture id에 대해 두 개의 JSON 파일이 생성됩니다.

```text
lecture_1937322_detail.json
lecture_1937322_articles.json
```

## 현재 구현 범위(MVP)

현재 구현된 범위는 다음과 같습니다.

- `lecture_ids.json` 기반 lecture id 목록 관리
- 내부 API 2개 호출
- form-encoded POST 요청 전송
- API 요청 간 기본 1.5초 대기
- 쿠키 기반 인증 정보 전달
- API 응답 JSON 파싱
- `status != success` 응답 실패 처리
- id별 detail/articles JSON 파일 저장

현재 실제 테스트 결과, 샘플 lecture id `488935`, `1937322`에 대해 다음 파일이 정상 생성되었습니다.

```text
data/raw/lecture_488935_detail.json
data/raw/lecture_488935_articles.json
data/raw/lecture_1937322_detail.json
data/raw/lecture_1937322_articles.json
```

응답 구조는 다음과 같이 확인되었습니다.

| 파일 유형 | `result` 주요 key |
| --- | --- |
| `_detail.json` | `books`, `exam`, `review` |
| `_articles.json` | `articles` |

## 설치 방법

필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

현재 필요한 주요 패키지는 다음과 같습니다.

```text
requests
```

## 실행 방법

API 호출에는 로그인 세션 쿠키가 필요합니다. 쿠키는 코드에 직접 저장하지 않고 환경변수 `EVERYTIME_COOKIE`로 전달합니다.

PowerShell 예시:

```powershell
$env:EVERYTIME_COOKIE='브라우저에서_확인한_쿠키_전체_문자열'
python src\collect.py
```

실행이 성공하면 `data/raw/` 아래에 lecture id별 JSON 파일이 생성됩니다.

기본적으로 API 요청 사이에는 1.5초의 지연 시간이 적용됩니다. 이 값은 [src/config.py](src/config.py)의 `REQUEST_DELAY_SECONDS`에서 조정할 수 있습니다.

## 설정 변경

수정이 잦은 값은 [src/config.py](src/config.py)에 모아두었습니다. 각 변수 근처에는 `[수정 포인트]` 주석이 붙어 있어 어떤 값을 바꾸면 되는지 빠르게 확인할 수 있습니다.

주요 설정값은 다음과 같습니다.

| 설정값 | 설명 |
| --- | --- |
| `LECTURE_API_URL` | 강의 상세 데이터 API URL입니다. |
| `ARTICLE_LIST_API_URL` | 강의평가 글 목록 API URL입니다. |
| `COOKIE_ENV_NAME` | 쿠키를 읽어올 환경변수 이름입니다. |
| `DEFAULT_LIMIT` | 한 번에 요청할 데이터 개수입니다. |
| `DEFAULT_OFFSET` | 페이지네이션 시작 위치입니다. |
| `DEFAULT_SORT` | API 응답 정렬 기준입니다. |
| `REQUEST_DELAY_SECONDS` | API 요청 간 최소 대기 시간입니다. |
| `REQUEST_TIMEOUT_SECONDS` | 요청 응답을 기다리는 최대 시간입니다. |

## 예시 JSON 구조

강의 상세 데이터 응답 예시:

```json
{
  "status": "success",
  "result": {
    "books": [],
    "exam": {},
    "review": {}
  }
}
```

강의평가 글 목록 응답 예시:

```json
{
  "status": "success",
  "result": {
    "articles": [
      {
        "id": 5220356,
        "year": 2022,
        "semester": "2",
        "text": "강의평가 본문 예시",
        "rate": 5,
        "posvote": 0
      }
    ]
  }
}
```

실제 응답에는 서비스 내부 구조에 따라 더 많은 필드가 포함될 수 있습니다.

## 주의사항 및 한계

이 프로젝트는 무분별한 대량 크롤링을 지향하지 않습니다. 필요한 lecture id만 선별적으로 수집하고, 서버에 과도한 부하를 주지 않는 방식으로 사용하는 것을 전제로 합니다.

사용 시 다음 사항을 반드시 고려해야 합니다.

- 대상 서비스의 이용약관과 정책을 확인해야 합니다.
- 쿠키, 인증 토큰 등 민감 정보는 코드에 직접 저장하지 않습니다.
- 쿠키는 만료될 수 있으며, 만료 시 API 요청이 실패할 수 있습니다.
- 내부 API URL, payload key, 응답 구조는 서비스 업데이트에 따라 변경될 수 있습니다.
- 요청 간 대기 시간, 재시도 정책, 실패 로그 관리는 향후 보강이 필요합니다.
- 수집 데이터에 개인정보 또는 민감한 내용이 포함될 경우 저장, 처리, 공유에 주의해야 합니다.

## 향후 확장 계획

향후에는 다음 기능을 추가할 수 있습니다.

- 실패 요청 재시도 및 실패 로그 저장
- 기존 JSON 파일이 있는 경우 skip하는 옵션
- offset 변경을 통한 페이지네이션 수집
- 여러 JSON 파일 병합
- 중복 article 제거
- 누락 필드 검증
- 강의, 교수, 평가 항목 간 그래프 데이터 생성
- GNN 학습용 node, edge 데이터셋 생성
- AI 기반 강의 추천 모델 학습 데이터 구축

## 개발 방향

이 프로젝트는 단순히 데이터를 많이 모으는 것보다, 이후 분석과 학습에 사용할 수 있는 구조화된 데이터를 안정적으로 확보하는 데 초점을 둡니다.

따라서 다음 원칙을 따릅니다.

- 필요한 데이터만 선별적으로 수집합니다.
- HTML 파싱보다 JSON 기반 구조화 데이터를 우선합니다.
- 원본 데이터와 전처리 데이터를 분리합니다.
- API 요청, 저장, 병합, 전처리 로직을 모듈화합니다.
- 추천 시스템과 GNN 분석으로 확장 가능한 데이터 구조를 유지합니다.

## 라이선스

아직 라이선스가 정해지지 않았다면, 프로젝트 공개 범위와 데이터 사용 정책을 먼저 결정한 뒤 라이선스를 선택하는 것을 권장합니다.
