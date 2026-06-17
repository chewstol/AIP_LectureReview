# Persona Evaluation

`evaluation/`은 `persona_maker`에서 생성한 페르소나가 기존 추천 모델의 feature space 안에서 얼마나 잘 동작하는지 평가하기 위한 코드와 결과를 담고 있습니다.

평가는 실제 사용자 로그 기반 검증이 아니라, 생성된 페르소나의 `selected_reviews`를 기준으로 한 사후 점검입니다. 각 페르소나가 선택한 강의평가의 별점과 모델 예측값을 비교하고, 해당 강의들이 페르소나 추천 순위 상위권에 들어오는지 확인합니다.

## 파일 구조

```text
evaluation/
├── README.md
├── scripts/
│   ├── convert_generated_personas.py
│   └── evaluate_generated_personas.py
└── data/
    ├── generated_personas.json
    ├── persona_weights/
    │   ├── manifest_initial_preference_vector.csv
    │   └── persona_{id}_{preset}_initial_preference_vector.json
    ├── persona_weights_aggregated/
    │   ├── manifest_aggregated_review_vector.csv
    │   └── persona_{id}_{preset}_aggregated_review_vector.json
    └── persona_evaluation/
        ├── persona_metrics.csv
        ├── review_predictions.csv
        └── summary.md
```

## 입력 데이터

- `data/generated_personas.json`: 생성된 페르소나 목록입니다.
- `model/data/model/lecture_nodes_with_text.csv`: 기존 추천 모델이 사용하는 강의 feature 데이터입니다.
- `selected_reviews[].rate`: 평가에서 정답처럼 사용하는 개별 강의평가 별점입니다.

`lecture_nodes_with_text.csv`는 현재 `evaluation/` 내부가 아니라 저장소의 `model/` 폴더 아래에 있습니다. 따라서 루트 디렉토리에서 실행할 때는 `--nodes` 옵션으로 경로를 명시합니다.

## 실행 방법

저장소 루트에서 실행합니다.

```powershell
python evaluation\scripts\convert_generated_personas.py `
  --personas evaluation\data\generated_personas.json `
  --nodes model\data\model\lecture_nodes_with_text.csv `
  --out-dir evaluation\data\persona_weights `
  --vector-field initial_preference_vector
```

```powershell
python evaluation\scripts\convert_generated_personas.py `
  --personas evaluation\data\generated_personas.json `
  --nodes model\data\model\lecture_nodes_with_text.csv `
  --out-dir evaluation\data\persona_weights_aggregated `
  --vector-field aggregated_review_vector
```

```powershell
python evaluation\scripts\evaluate_generated_personas.py `
  --personas evaluation\data\generated_personas.json `
  --nodes model\data\model\lecture_nodes_with_text.csv `
  --out-dir evaluation\data\persona_evaluation
```

## 스크립트 설명

- `convert_generated_personas.py`: `generated_personas.json` 안의 페르소나 벡터를 추천 엔진에서 사용할 수 있는 weight JSON 파일로 변환합니다.
- `evaluate_generated_personas.py`: `initial_preference_vector`와 `aggregated_review_vector`를 각각 평가하고, 페르소나별 정확도와 추천 순위 지표를 계산합니다.

## 평가 지표

- `MAE`, `RMSE`: 모델이 예측한 5점 척도 품질 점수와 실제 선택 리뷰 별점의 차이입니다.
- `Bias`: 예측값이 실제 별점보다 높은지 낮은지 보는 평균 오차입니다.
- `Within 0.5`, `Within 1.0`: 예측 오차가 각각 0.5점, 1.0점 이내인 비율입니다.
- `Hit@10`, `Hit@30`, `Hit@100`: 페르소나가 선택한 강의가 추천 순위 Top-K 안에 포함된 비율입니다.
- `Median rank`, `Mean rank`: 선택된 강의들의 추천 순위 중앙값과 평균값입니다.

## 산출물

- `data/persona_weights/`: `initial_preference_vector` 기반 weight 파일입니다.
- `data/persona_weights_aggregated/`: `aggregated_review_vector` 기반 weight 파일입니다.
- `data/persona_evaluation/persona_metrics.csv`: 페르소나 단위 평가 지표입니다.
- `data/persona_evaluation/review_predictions.csv`: 개별 선택 리뷰 단위 예측 결과입니다.
- `data/persona_evaluation/summary.md`: 전체 결과 요약과 해석입니다.

## 해석 시 주의사항

이 평가는 생성된 페르소나가 기존 추천 모델과 얼마나 잘 맞는지 확인하는 내부 검증입니다. 실제 학생 이력 기반의 held-out 추천 평가가 아니므로, 개인화 추천 성능을 최종적으로 증명하는 지표로 보기는 어렵습니다.

특히 개별 리뷰 별점은 강의 평균 별점보다 변동성이 크기 때문에, 별점 예측 오차는 기존 cross-validation 결과보다 엄격하게 해석해야 합니다. `Hit@K`는 페르소나 벡터가 선택된 리뷰의 강의를 추천 상위권에 올리는지를 보는 보조 지표로 사용합니다.
