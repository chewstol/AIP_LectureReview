# Lecture Recommendation Model

`model/`은 정규화된 강의평가 데이터를 사용해 강의 feature를 만들고, 평균 평점 예측 모델과 Top-K 추천 방식을 실험한 코드와 결과를 담고 있습니다.

이 폴더는 `crawler/`에서 만들어진 정규화 데이터와 feature 데이터를 기반으로 더 다양한 모델 비교, 하이퍼파라미터 탐색, 추천 결과 생성을 수행합니다.

## 파일 구조

```text
model/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── build_lecture_nodes.py
│   ├── build_text_features.py
│   ├── run_full_cv_experiment.py
│   ├── run_extended_model_experiment.py
│   ├── run_kobert_all_models.py
│   ├── run_transformer_tabular_experiment.py
│   ├── recommend_topk.py
│   └── make_*_visuals.py
└── data/
    ├── model/
    │   ├── lecture_nodes.csv
    │   ├── lecture_nodes_with_text.csv
    │   ├── lecture_text_features.csv
    │   ├── lecture_top_keywords.csv
    │   └── lecture_kobert_embeddings.npz
    ├── experiments/
    └── recommendations/
```

## 모델 입력

주요 입력 파일은 `data/model/lecture_nodes_with_text.csv`입니다.

강의는 다음 feature로 표현됩니다.

- 구조화 feature: 과제, 팀플, 학점, 출석, 시험 부담 관련 점수
- 텍스트 feature: 강의평가 문장에서 계산한 TF-IDF 기반 주제 feature
- 실험별 확장 feature: KoBERT embedding, leakage-safe feature subset 등

예측 target은 `rating_average_norm`이며, 5점 만점 평균 별점을 0~1 범위로 정규화한 값입니다.

```text
rating_average_norm = rating_average / 5
```

## 주요 실험

- `full_cv/`: 전체 데이터 5-fold cross-validation 결과입니다.
- `extended_models/`: Ridge, SVR, Random Forest, Extra Trees, Gradient Boosting 등 확장 모델 비교 결과입니다.
- `leakage_safe_legacy/`: 별점에서 직접 파생된 feature를 제외한 조건의 비교 결과입니다.
- `tfidf16_final/`: 최종 발표용 TF-IDF 16개 feature 구성 실험입니다.
- `tfidf16_kernel_ridge_tuning/`: Kernel Ridge 하이퍼파라미터 탐색 결과입니다.
- `kobert_all_models/`, `transformer_tabular/`, `kobert_mlp/`: KoBERT embedding 결합 실험 결과입니다.
- `final_six_models/`: 발표용 최종 6개 모델 비교 결과입니다.

## 실행 방법

`model/` 폴더에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

기본 feature와 전체 cross-validation 결과를 다시 만들려면 다음 명령을 실행합니다.

```powershell
python scripts\build_lecture_nodes.py
python scripts\build_text_features.py
python scripts\run_full_cv_experiment.py
python scripts\make_cv_visuals.py
```

확장 모델 비교와 최종 발표용 결과는 다음 스크립트들로 생성합니다.

```powershell
python scripts\run_extended_model_experiment.py
python scripts\run_leakage_safe_legacy_models.py
python scripts\make_extended_model_visuals.py
python scripts\run_kobert_all_models.py
python scripts\make_kobert_all_models_visual.py
python scripts\make_six_model_comparison.py
```

Top-K 추천 결과는 다음과 같이 생성합니다.

```powershell
python scripts\recommend_topk.py --preset low_workload --top-k 10
```

직접 선호 가중치를 입력할 수도 있습니다.

```powershell
python scripts\recommend_topk.py --weights-json '{ "assignment_low_score": 1.0, "exam_light_score": 0.8 }'
```

## 추천 방식

추천 점수는 선호 벡터와 강의 벡터의 유사도, 그리고 모델이 예측한 강의 품질 점수를 결합합니다.

```text
recommendation_score = 0.7 * preference_similarity + 0.3 * predicted_quality
```

- `preference_similarity`: 사용자 또는 페르소나 선호 벡터와 강의 feature vector의 cosine similarity입니다.
- `predicted_quality`: 평점 예측 모델이 계산한 강의 품질 점수입니다.
- `top-k`: 추천 결과로 반환할 강의 수입니다.

## 산출물

- `data/model/lecture_nodes_with_text.csv`: 주요 모델 입력 데이터입니다.
- `data/experiments/*`: 모델별 실험 결과, fold별 지표, 시각화 파일입니다.
- `data/recommendations/*`: preset별 추천 결과와 요약입니다.

## 해석 시 주의사항

현재 평가는 평균 평점 예측과 feature 기반 추천에 초점을 둡니다. 학생별 수강 이력이나 개인별 선호 로그가 없기 때문에, collaborative filtering이나 실제 개인화 추천 정확도를 직접 검증한 결과는 아닙니다.

또한 일부 텍스트/감성 feature는 기존 리뷰 별점에서 파생된 정보와 가까울 수 있어 target leakage 가능성을 함께 고려해야 합니다. 발표나 보고서에서는 leakage-safe 실험 결과와 함께 해석하는 것이 좋습니다.
