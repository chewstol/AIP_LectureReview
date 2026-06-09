# AIP Lecture Recommendation Experiment Share

조원 검토용 공유 패키지입니다. 코드, 정규화 데이터, 모델 입력 데이터, 실험 결과, Top-K 추천 결과를 포함합니다.

## 먼저 볼 파일

1. `data/experiments/full_cv/cv_presentation_summary.md`
   - 06/08 발표 요구사항 기준 요약
   - 전체 데이터 cross-validation, 하이퍼파라미터 탐색, 성능 비교, qualitative analysis 정리

2. `data/experiments/full_cv/cv_summary.csv`
   - 5-fold cross-validation 결과 요약
   - 모델별 MSE, RMSE, MAE 평균 및 표준편차

3. `data/recommendations/README.md`
   - 사용자 선호 벡터 기반 Top-K 추천 방식 설명
   - 선호 시나리오별 추천 결과 파일 위치

4. `data/recommendations/topk_*_10.csv`
   - 선호 시나리오별 Top-10 추천 결과
   - 현재 데이터에는 과목명/교수명 metadata가 없어 `lecture_id` 중심으로 표시됨

5. `data/experiments/full_cv/*.svg`
   - 발표용 그래프

## 주요 결과

전체 데이터 기준:

- 강의 노드: 753개
- Feature: 31개
- Target: `rating_average_norm`
- Cross-validation: 5-fold

Best model:

```text
Ridge Regression alpha=10
CV MSE  0.00319527
CV RMSE 0.05641645
CV MAE  0.04178956
```

## Top-K 추천 구조

추천 점수:

```text
recommendation_score = 0.7 * preference_similarity + 0.3 * predicted_quality
```

- `preference_similarity`: 사용자 선호 벡터와 강의 벡터의 cosine similarity
- `predicted_quality`: Ridge regression이 예측한 강의 품질 점수
- `rating_average`: 실제 평균 별점이며 결과 해석용으로 함께 출력

실행 예시:

```powershell
python scripts\recommend_topk.py --preset low_workload --top-k 10
```

## 폴더 구조

```text
scripts/
  build_lecture_nodes.py
  build_text_features.py
  run_small_portion_experiment.py
  run_full_cv_experiment.py
  recommend_topk.py
  make_cv_visuals.py

crawler_src/
  api_client.py
  collect.py
  config.py

data/
  normalized/
    lecture_articles.csv
    lecture_details.csv
    exam_question_types.csv
    books.csv

  model/
    lecture_nodes.csv
    lecture_nodes_with_text.csv
    lecture_text_features.csv
    lecture_top_keywords.csv

  experiments/
    small_portion/
    full_data/
    full_cv/

  recommendations/
    topk_low_workload_10.csv
    topk_learning_quality_10.csv
    topk_exam_light_10.csv
    topk_no_team_project_10.csv
    topk_challenging_but_good_10.csv
```

## 재현 방법

Python 실행 파일은 각자 환경에 맞게 `python`으로 바꾸면 됩니다.

```powershell
python scripts\build_lecture_nodes.py
python scripts\build_text_features.py
python scripts\run_full_cv_experiment.py
python scripts\make_cv_visuals.py
python scripts\recommend_topk.py --preset low_workload --top-k 10
```

## 포함하지 않은 것

원본 raw JSON 전체는 공유 패키지에 포함하지 않았습니다.

이유:

- raw 데이터 용량이 크고 민감할 수 있음
- 실험 검토에는 정규화된 CSV와 모델 입력 데이터만으로 충분함
- 필요하면 원본 raw는 별도 공유 필요

## 현재 한계

- 과목명/교수명 metadata가 없음
- 학생별 수강/평가 이력 데이터가 없음
- 현재 검증 target은 개인화 추천 정확도가 아니라 평균 평점 예측
- 실제 개인화 추천 검증에는 Hit@K, NDCG@K 같은 ranking metric과 학생 이력 데이터가 필요함
