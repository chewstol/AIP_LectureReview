# AIP Lecture Review Recommendation

강의평가 데이터를 기반으로 강의 feature를 만들고, 평균 평점 예측 모델과 Top-K 추천 방식을 실험한 프로젝트입니다. 추가로 생성형 페르소나를 만들어 추천 입력으로 사용하고, 해당 페르소나가 기존 추천 모델 안에서 얼마나 잘 동작하는지 평가합니다.

이 저장소는 역할별로 네 개의 하위 폴더로 나뉩니다.

```text
AIP_LectureReview/
├── crawler/
├── model/
├── persona_maker/
└── evaluation/
```

## 전체 흐름

```text
crawler/
  원본 강의평가 수집 및 정규화
        ↓
model/
  강의 feature 생성, 평점 예측 모델 실험, Top-K 추천
        ↓
persona_maker/
  리뷰와 feature를 이용한 실험용 페르소나 생성
        ↓
evaluation/
  생성된 페르소나의 예측 오차와 추천 순위 평가
```

## 폴더별 역할

### `crawler/`

강의평가 데이터를 수집하고 정규화하는 파이프라인입니다. 원본 API 응답을 분석 가능한 CSV와 기본 feature 데이터로 변환합니다.

주요 산출물:

- `crawler/data/normalized/lecture_articles.csv`
- `crawler/data/normalized/lecture_details.csv`
- `crawler/data/model/lecture_nodes_with_text.csv`

자세한 내용은 `crawler/README.md`를 참고합니다.

### `model/`

정규화된 데이터를 이용해 강의 feature를 만들고, 평균 평점 예측 모델과 추천 방식을 실험합니다. Ridge, Kernel Ridge, tree 기반 모델, KoBERT 결합 실험 등이 포함되어 있습니다.

주요 산출물:

- `model/data/model/lecture_nodes_with_text.csv`
- `model/data/experiments/`
- `model/data/recommendations/`

자세한 내용은 `model/README.md`를 참고합니다.

### `persona_maker/`

강의평가 리뷰와 강의 feature를 바탕으로 실험용 페르소나를 생성합니다. 페르소나는 추천 입력으로 사용할 선호 벡터와 선택 리뷰 목록을 포함합니다.

주요 산출물:

- `persona_maker/outputs/generated_personas.json`
- `persona_maker/outputs/generated_personas_simple.json`

자세한 내용은 `persona_maker/README.md`를 참고합니다.

### `evaluation/`

생성된 페르소나가 기존 추천 모델 feature space에서 얼마나 잘 맞는지 평가합니다. 별점 예측 오차와 Hit@K 같은 추천 순위 지표를 계산합니다.

주요 산출물:

- `evaluation/data/persona_evaluation/persona_metrics.csv`
- `evaluation/data/persona_evaluation/review_predictions.csv`
- `evaluation/data/persona_evaluation/summary.md`

자세한 내용은 `evaluation/README.md`를 참고합니다.

## 실행 순서

각 폴더는 독립적으로 실행할 수 있지만, 전체 재현 흐름은 다음 순서를 따릅니다.

```powershell
cd crawler
pip install -r requirements.txt
python scripts\normalize_raw.py
python scripts\build_lecture_nodes.py
python scripts\build_text_features.py
```

```powershell
cd ..\model
pip install -r requirements.txt
python scripts\run_full_cv_experiment.py
python scripts\recommend_topk.py --preset low_workload --top-k 10
```

```powershell
cd ..\persona_maker
python src\generate_review_mention_vectors.py
python src\generate_review_calibrated_vectors.py
python src\generate_personas_from_reviews.py
```

```powershell
cd ..
python evaluation\scripts\evaluate_generated_personas.py `
  --personas evaluation\data\generated_personas.json `
  --nodes model\data\model\lecture_nodes_with_text.csv `
  --out-dir evaluation\data\persona_evaluation
```

## 핵심 결과물

- 강의 feature 데이터: `model/data/model/lecture_nodes_with_text.csv`
- 모델 실험 결과: `model/data/experiments/`
- 추천 결과: `model/data/recommendations/`
- 생성 페르소나: `persona_maker/outputs/generated_personas.json`
- 페르소나 평가 요약: `evaluation/data/persona_evaluation/summary.md`

## 해석 시 주의사항

현재 프로젝트는 평균 평점 예측과 feature 기반 추천 실험에 초점을 둡니다. 학생별 수강 이력이나 실제 사용자 interaction 로그가 없기 때문에, 개인화 추천 성능을 최종적으로 검증한 결과로 해석하기는 어렵습니다.

또한 일부 텍스트/감성 feature는 기존 리뷰 별점에서 파생된 정보와 가까울 수 있어 target leakage 가능성을 함께 고려해야 합니다. 모델 성능을 설명할 때는 leakage-safe 실험 결과와 실제 사용자 검증 한계를 같이 언급하는 것이 좋습니다.

원본 강의평가 JSON과 민감할 수 있는 데이터는 공개/공유용 산출물에 포함하지 않는 것을 전제로 합니다.
