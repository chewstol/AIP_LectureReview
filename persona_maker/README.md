# Persona Maker

`persona_maker/`는 강의평가 feature와 리뷰 텍스트를 이용해 추천 실험에 사용할 페르소나 데이터를 생성하는 코드와 산출물을 담고 있습니다.

생성된 페르소나는 `model/`의 추천 엔진 또는 `evaluation/`의 평가 스크립트에서 사용할 수 있는 선호 벡터와 선택 리뷰 목록을 포함합니다.

## 파일 구조

```text
persona_maker/
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
│       ├── lecture_{lecture_id}_articles.json
│       └── lecture_{lecture_id}_detail.json
└── outputs/
    ├── generated_personas.json
    └── generated_personas_simple.json
```

## 생성 흐름

1. `generate_review_mention_vectors.py`
   - 강의평가 텍스트에서 과제, 팀플, 학점, 출석, 시험, 강의력 관련 언급 여부를 추출합니다.
   - 결과는 `data/processed/review_mention_vectors.json`에 저장됩니다.

2. `generate_review_calibrated_vectors.py`
   - 리뷰의 mention vector와 강의 feature를 결합합니다.
   - 리뷰에서 언급된 항목에 대해서만 강의 feature 값을 남기고, 나머지는 0으로 보정합니다.
   - 결과는 `data/processed/review_calibrated_vectors.json`과 `.csv`로 저장됩니다.

3. `generate_personas_from_reviews.py`
   - preset 기반 초기 선호 벡터를 만들고, 해당 벡터와 가까운 리뷰를 샘플링해 페르소나를 생성합니다.
   - 결과는 `outputs/generated_personas.json`에 저장됩니다.

## 실행 방법

`persona_maker/` 폴더에서 실행합니다.

```powershell
python src\generate_review_mention_vectors.py
python src\generate_review_calibrated_vectors.py
python src\generate_personas_from_reviews.py
```

간소화된 페르소나 파일만 생성하려면 다음 옵션을 사용합니다.

```powershell
python src\generate_personas_from_reviews.py --simple
```

각 스크립트는 파일 위치를 `persona_maker/` 기준으로 계산하므로, 저장소 루트에서 실행하는 것보다 `persona_maker/` 안에서 실행하는 편이 가장 명확합니다.

## 사용 feature

페르소나 벡터는 16개 feature를 사용합니다.

구조화 feature 10개:

- `assignment_low_score`, `assignment_high_score`
- `teamwork_low_score`, `teamwork_high_score`
- `grading_generous_score`, `grading_strict_score`
- `attendance_light_score`, `attendance_strict_score`
- `exam_light_score`, `exam_heavy_score`

텍스트 feature 6개:

- `text_assignment_tfidf`
- `text_teamwork_tfidf`
- `text_grading_tfidf`
- `text_attendance_tfidf`
- `text_exam_tfidf`
- `text_teaching_tfidf`

## Preset

현재 페르소나는 다음 preset을 기반으로 생성됩니다.

- `low_workload`: 과제, 팀플, 출석, 시험 부담이 낮은 강의를 선호합니다.
- `learning_quality`: 강의 설명과 학습 품질이 높은 강의를 선호합니다.
- `grade_focused`: 학점 부담이 낮거나 관대한 평가를 선호합니다.
- `balanced`: 특정 요소에 크게 치우치지 않은 균형형 선호입니다.

각 preset에는 작은 noise가 더해져 여러 개의 페르소나가 생성됩니다.

## 산출물

### `outputs/generated_personas.json`

전체 페르소나 정보입니다. 초기 선호 벡터, 선택된 리뷰, 리뷰별 유사도, mention vector, calibrated vector, 집계된 리뷰 벡터를 포함합니다.

주요 필드:

- `persona_id`: 페르소나 고유 번호입니다.
- `preset_name`: 페르소나 생성에 사용된 preset 이름입니다.
- `initial_preference_vector`: preset과 noise로 만든 초기 선호 벡터입니다.
- `selected_reviews`: 페르소나를 구성하는 선택 리뷰 목록입니다.
- `similarity`: 초기 선호 벡터와 해당 리뷰 vector 사이의 cosine similarity입니다.
- `mention_vector`: 리뷰 텍스트에서 각 항목이 언급되었는지 나타내는 0/1 벡터입니다.
- `calibrated_vector`: 언급된 항목에 대해서만 강의 feature 값을 남긴 리뷰 단위 벡터입니다.
- `aggregated_review_vector`: 선택 리뷰들의 calibrated vector를 평균낸 벡터입니다.

### `outputs/generated_personas_simple.json`

페르소나 사용에 필요한 최소 정보만 담은 축약 파일입니다. `--simple` 옵션으로 생성합니다.

포함 필드:

- `persona_id`
- `preset_name`
- `initial_preference_vector`
- `selected_reviews[].text`

## 다른 폴더와의 관계

- `model/`: 생성된 페르소나 선호 벡터를 추천 입력으로 사용할 수 있습니다.
- `evaluation/`: `generated_personas.json`을 입력으로 받아 페르소나의 추천 적합성과 예측 오차를 평가합니다.

## 주의사항

이 폴더의 페르소나는 실제 학생 로그에서 직접 학습된 사용자가 아니라, 강의평가 리뷰와 preset 선호를 조합해 만든 실험용 페르소나입니다. 따라서 최종 개인화 추천 성능을 주장할 때는 `evaluation/` 결과와 실제 사용자 검증 한계를 함께 설명해야 합니다.
