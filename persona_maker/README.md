# PersonaMaker

## 프로젝트 목표

PersonaMaker는 비슷한 성향을 가진 강의평가를 묶어서 페르소나를 생성하는 코드 작성을 목표로 합니다.

입력 벡터와 기존 강의평가 데이터의 유사도를 계산하고, 가장 유사한 강의평가들을 기반으로 대표적인 성향 또는 페르소나를 도출하는 데 활용할 수 있습니다.

## 파일 구조

```text
PersonaMaker/
├── README.md
├── src/
│   ├── generate_review_mention_vectors.py
│   ├── generate_review_calibrated_vectors.py
│   └── generate_personas_from_reviews.py
├── data/
│   ├── csv/
│   │   ├── lecture_articles.csv
│   │   ├── lecture_nodes.csv
│   │   ├── lecture_nodes_with_text.csv
│   │   ├── lecture_text_features.csv
│   │   └── lecture_top_keywords.csv
│   ├── processed/
│   │   ├── review_mention_vectors.json
│   │   ├── review_calibrated_vectors.json
│   │   └── review_calibrated_vectors.csv
│   └── raw/
│       └── 원본 강의평가 JSON 파일들
└── outputs/
    ├── generated_personas.json
    └── generated_personas_simple.json
```

- `src/`: 데이터 변환과 페르소나 생성을 수행하는 Python 실행 코드
- `data/csv/`: 정리된 입력 CSV 파일
- `data/raw/`: 원본 JSON 데이터
- `data/processed/`: 리뷰 mention vector, 2차 보정 벡터 등 중간 산출물
- `outputs/`: 최종 페르소나 JSON 산출물

## 실행 방법

프로젝트 루트에서 다음 순서로 실행합니다.

```bash
python src/generate_review_mention_vectors.py
python src/generate_review_calibrated_vectors.py
python src/generate_personas_from_reviews.py
```

간소화된 페르소나 파일만 생성하려면 다음 옵션을 사용합니다.

```bash
python src/generate_personas_from_reviews.py --simple
```

각 스크립트는 파일 위치가 정리된 뒤에도 동작하도록 프로젝트 루트 기준 경로를 자동으로 계산합니다.

## Input

- `n`개 Feature에 대한 정규화되지 않은 벡터
- 각 Feature는 강의평가의 성향, 특징, 평가 요소 등을 수치로 표현한 값입니다.
- 입력 벡터는 정규화되지 않은 상태로 들어오므로, 필요한 경우 코드 내부에서 정규화 또는 스케일링 과정을 수행할 수 있습니다.
- CSV 파일에 포함된 Feature 중 아래에 명시된 Feature만 사용합니다.

### 사용할 Feature

#### 정형 Feature 10개

| Feature | 설명 |
| --- | --- |
| `assignment_low_score` | 과제 부담이 낮은 정도 |
| `assignment_high_score` | 과제 부담이 높은 정도 |
| `teamwork_low_score` | 팀플이 적은 정도 |
| `teamwork_high_score` | 팀플이 많은 정도 |
| `grading_generous_score` | 학점을 후하게 주는 정도 |
| `grading_strict_score` | 학점이 엄격한 정도 |
| `attendance_light_score` | 출석 부담이 낮은 정도 |
| `attendance_strict_score` | 출석이 엄격한 정도 |
| `exam_light_score` | 시험 부담이 낮은 정도 |
| `exam_heavy_score` | 시험 부담이 높은 정도 |

#### 텍스트 Feature 6개

| Feature | 설명 |
| --- | --- |
| `text_assignment_tfidf` | 과제 관련 표현의 TF-IDF |
| `text_teamwork_tfidf` | 팀플 관련 표현의 TF-IDF |
| `text_grading_tfidf` | 학점 관련 표현의 TF-IDF |
| `text_attendance_tfidf` | 출석 관련 표현의 TF-IDF |
| `text_exam_tfidf` | 시험 관련 표현의 TF-IDF |
| `text_teaching_tfidf` | 강의력·설명 관련 표현의 TF-IDF |

텍스트 Feature는 강의평가 원문에서 특정 주제와 관련된 표현이 얼마나 중요하게 나타나는지를 수치화한 값입니다. 각 값은 TF-IDF 기반으로 계산되며, 특정 강의평가에서 해당 주제의 단어 또는 표현이 자주 등장하면서도 전체 강의평가에서는 상대적으로 구분력 있게 사용될수록 높은 값을 가질 수 있습니다.

- `text_assignment_tfidf`: 과제, 레포트, 제출, 숙제 등 과제 부담과 관련된 표현의 중요도
- `text_teamwork_tfidf`: 팀플, 조별과제, 발표 조, 협업 등 팀워크 관련 표현의 중요도
- `text_grading_tfidf`: 학점, 점수, 성적, 커브, 후함, 엄격함 등 평가 방식과 관련된 표현의 중요도
- `text_attendance_tfidf`: 출석, 결석, 지각, 출결 확인 등 출석 관리와 관련된 표현의 중요도
- `text_exam_tfidf`: 시험, 중간고사, 기말고사, 퀴즈, 족보 등 시험 부담과 관련된 표현의 중요도
- `text_teaching_tfidf`: 설명, 강의력, 전달력, 이해, 교수 방식 등 수업 진행과 관련된 표현의 중요도

## Output

- 강의평가 목록 중에서 입력 벡터와 가장 유사한 `top-k`개의 강의평가를 담은 JSON file
- 출력 JSON file은 유사도 기준으로 선택된 강의평가 목록을 포함합니다.
- 각 결과 항목에는 구현 방식에 따라 강의평가 정보, 유사도 점수, 관련 메타데이터 등을 포함할 수 있습니다.

### `outputs/generated_personas.json`

`generated_personas.json`은 preset 기반 랜덤 선호도 벡터와, 해당 선호도 벡터에 가까운 강의평가 리뷰들을 묶은 전체 페르소나 데이터입니다. 유사도, mention vector, calibrated vector까지 포함하므로 페르소나 생성 과정을 분석하거나 디버깅할 때 사용할 수 있습니다.

주요 구조는 다음과 같습니다.

```json
[
  {
    "persona_id": 1,
    "preset_name": "low_workload",
    "initial_preference_vector": {
      "assignment_low_score": 0.88,
      "assignment_high_score": 0.12,
      "text_teaching_tfidf": 0.3
    },
    "top_k": 30,
    "sample_n": 10,
    "selected_reviews": [
      {
        "lecture_id": 103298,
        "article_id": 2131858,
        "text": "강의평가 원문",
        "rate": 3.0,
        "similarity": 0.8231,
        "mention_vector": {
          "assignment": 1,
          "teamwork": 0,
          "grading": 0,
          "attendance": 0,
          "exam": 1,
          "teaching": 1
        },
        "calibrated_vector": {
          "assignment_low_score": 0.93,
          "assignment_high_score": 0.07,
          "text_teaching_tfidf": 0.12
        }
      }
    ],
    "aggregated_review_vector": {
      "assignment_low_score": 0.71,
      "assignment_high_score": 0.12,
      "text_teaching_tfidf": 0.08
    }
  }
]
```

- `persona_id`: 생성된 페르소나의 고유 번호
- `preset_name`: 페르소나 생성에 사용된 preset 이름
- `initial_preference_vector`: preset에 랜덤 노이즈를 더해 생성한 초기 선호도 벡터
- `top_k`: 초기 선호도 벡터와 가장 가까운 후보 리뷰 수
- `sample_n`: `top_k` 후보 중 최종 선택한 리뷰 수
- `selected_reviews`: 페르소나를 구성하는 강의평가 리뷰 목록
- `similarity`: 초기 선호도 벡터와 해당 리뷰의 `calibrated_vector` 사이의 cosine similarity
- `mention_vector`: 리뷰 텍스트에서 각 요소가 언급되었는지 나타내는 0 또는 1 벡터
- `calibrated_vector`: 리뷰에서 언급된 요소에 대해서만 강의 feature 값을 남긴 2차 벡터
- `aggregated_review_vector`: 선택된 리뷰들의 `calibrated_vector`를 feature별 평균낸 벡터

### `outputs/generated_personas_simple.json`

`generated_personas_simple.json`은 페르소나 활용에 필요한 최소 정보만 남긴 축약 버전입니다. `generate_personas_from_reviews.py` 실행 시 `--simple` 옵션을 사용하면 생성됩니다.

포함되는 정보는 다음 네 가지입니다.

- `persona_id`
- `preset_name`
- `initial_preference_vector`
- `selected_reviews`의 `text`

주요 구조는 다음과 같습니다.

```json
[
  {
    "persona_id": 1,
    "preset_name": "low_workload",
    "initial_preference_vector": {
      "assignment_low_score": 0.88,
      "assignment_high_score": 0.12,
      "text_teaching_tfidf": 0.3
    },
    "selected_reviews": [
      {
        "text": "강의평가 원문"
      }
    ]
  }
]
```

이 파일은 유사도 점수나 중간 벡터 정보 없이, 생성된 페르소나의 선호도와 실제 강의평가 텍스트만 활용하고 싶을 때 사용합니다.

## 코드 작성 규칙

Python 파일에서 함수를 정의할 때는 함수 정의 한 줄 위에 함수의 역할을 설명하는 짧은 한글 주석을 작성합니다.

예시:

```python
# 입력 벡터와 강의평가 벡터 간 유사도를 계산한다.
def calculate_similarity(input_vector, review_vector):
    pass
```
