# AI-based Lecture Review Analysis and Recommendation

강의평 데이터를 정형 및 텍스트 feature로 변환하고, 평균 평점 예측과 사용자 선호 기반 Top-K 강의 추천을 실험한 프로젝트입니다.

## Project Overview

프로젝트의 주요 흐름은 다음과 같습니다.

1. 강의평 raw JSON 정규화
2. 리뷰 수가 적은 강의 제외
3. 강의별 31차원 feature vector 생성
4. 평균 평점 예측 모델 비교
5. 5-fold cross-validation 기반 하이퍼파라미터 탐색
6. 사용자 persona와 강의 vector의 유사도를 이용한 Top-K 추천

원본 강의평과 사용자 식별 가능성이 있는 데이터는 저장소에 포함하지 않습니다.

## Lecture Vector

각 강의는 총 31개의 feature로 표현됩니다.

- 정형 feature 10개
  - 과제 적음/많음
  - 팀플 적음/많음
  - 학점 후함/엄격함
  - 출석 부담 낮음/높음
  - 시험 부담 낮음/높음
- 텍스트 TF-IDF feature 18개
  - 과제, 시험, 팀플, 출석, 학점, 강의력 카테고리
  - 카테고리별 전체/긍정/부정 TF-IDF
- 감성 비율 feature 3개
  - 긍정, 중립, 부정 리뷰 비율

감성 라벨은 기존 리뷰 별점을 이용한 weak labeling입니다.

```text
4~5점: positive
3점: neutral
1~2점: negative
```

## Rating Prediction

예측 target은 5점 만점 평균 별점을 0~1로 정규화한 값입니다.

```text
rating_average_norm = rating_average / 5
```

비교 모델:

- Train mean baseline
- Ridge regression
- Content-based KNN
- Graph-augmented MLP

전체 753개 강의에 대해 5-fold cross-validation을 수행한 결과, Ridge regression이 가장 좋은 성능을 보였습니다.

```text
Best model: Ridge Regression, alpha=10
CV MSE:  0.00319527
CV RMSE: 0.05641645
CV MAE:  0.04178956
```

정규화 값을 5점 척도로 환산하면 RMSE는 약 0.28점, MAE는 약 0.21점입니다.

### Extended Model Comparison

작은 표형 데이터에 적합한 모델을 추가로 비교했습니다.

- RBF-SVR
- Kernel Ridge-RBF
- Random Forest
- Extra Trees
- Gradient Boosting
- Histogram Gradient Boosting
- Stacking Ensemble

31개 feature 전체에서는 Ridge가 여전히 가장 좋은 MSE를 보였습니다. 별점으로 생성한 감성 feature를 제외한 16개 `leakage_safe` feature에서는 Kernel Ridge-RBF가 가장 좋은 결과를 냈지만, Ridge 대비 MSE 개선은 약 0.9%로 근소했습니다.

```text
Leakage-safe best model: Kernel Ridge-RBF
CV MSE:  0.00722578
CV RMSE: 0.08430563
CV MAE:  0.06168934
```

`text_positive_ratio`, `text_negative_ratio`, 긍정/부정 TF-IDF는 예측 target과 동일한 리뷰 별점으로 만들어졌기 때문에 평균 평점 예측에서는 target leakage를 일으킬 수 있습니다. 따라서 모델의 일반화 성능을 주장할 때는 leakage-safe 결과를 함께 보고해야 합니다.

기존 Content KNN과 Graph-augmented MLP도 동일한 16개 leakage-safe feature로 다시 평가했습니다.

```text
Content KNN (k=20)
CV RMSE: 0.08776978
CV MAE:  0.06463440

Graph-augmented MLP (hidden=16, lr=0.003, epochs=500)
CV RMSE: 0.20609465
CV MAE:  0.15810297
```

KNN은 Ridge 계열보다 조금 낮지만 비교적 안정적인 결과를 보였습니다. Graph-augmented MLP는 feature 유사도로 만든 인접 관계가 실제 학생 선호나 수강 관계를 나타내지 못해 성능이 크게 낮았습니다.

### KoBERT + Structured Features

강의평 13,979개를 frozen KoBERT encoder로 임베딩하고, 강의별 평균 임베딩을 정형 feature 10개와 결합했습니다. PCA는 leakage 방지를 위해 각 cross-validation fold의 train 데이터에서만 학습했습니다.

```text
Structured 10 only Ridge RMSE:            0.085304
KoBERT PCA32 only Ridge RMSE:             0.108486
Structured 10 + KoBERT PCA32 Ridge RMSE:  0.082837
```

결합 모델은 정형 feature만 사용한 모델보다 RMSE가 약 2.9% 개선됐고, 이전 leakage-safe Kernel Ridge보다 약 1.7% 개선됐습니다. XGBoost와 LightGBM도 실험했지만 RMSE 기준으로는 Ridge가 가장 좋았습니다.

동일한 `정형 10 + KoBERT PCA32` 입력으로 전체 모델군을 다시 비교한 결과는 다음과 같습니다.

```text
Ridge                       RMSE 0.082837
XGBoost                     RMSE 0.084969
Extra Trees                 RMSE 0.085564
Gradient Boosting           RMSE 0.085815
Histogram Gradient Boosting RMSE 0.085839
LightGBM                    RMSE 0.086318
Random Forest               RMSE 0.086617
Kernel Ridge-RBF            RMSE 0.087002
SVR-RBF                     RMSE 0.088387
Content KNN                 RMSE 0.106021
```

TF-IDF 기반 16개 feature에서는 Kernel Ridge가 가장 좋았지만, KoBERT 결합 feature에서는 일반 Ridge가 가장 좋았습니다. 최적 모델은 feature 표현에 따라 달라질 수 있음을 보여줍니다.

발표용 대표 모델 6개 비교에는 일반 MLP도 동일한 최종 feature로 추가했습니다.

```text
Ridge             RMSE 0.082837
Gradient Boosting RMSE 0.085815
Random Forest     RMSE 0.086617
SVR-RBF           RMSE 0.088387
Content KNN       RMSE 0.106021
MLP               RMSE 0.114041
```

MLP는 `(64, 32)` 은닉층과 early stopping을 적용했지만, 753개의 작은 데이터에서는 Ridge보다 낮은 성능을 보였습니다.

### TF-IDF Final Presentation Configuration

팀 최종 발표에서는 해석 가능한 persona feature 공간과의 일관성을 위해 `정형 10 + 카테고리 TF-IDF 6` 구성을 사용합니다.

```text
Kernel Ridge-RBF  RMSE 0.084306
Ridge             RMSE 0.084741
Random Forest     RMSE 0.086671
Gradient Boosting RMSE 0.087399
Content KNN       RMSE 0.087770
MLP               RMSE 0.088816
SVR-RBF           RMSE 0.090054
```

정밀 grid search 결과 최종 모델은 `Kernel Ridge-RBF(alpha=1e-6, gamma=1e-5)`이며, CV RMSE는 `0.084056`입니다. 기존 coarse-grid 설정인 `alpha=0.01, gamma=0.001`과의 차이는 작아 작은 alpha와 gamma 영역이 넓은 안정 구간을 형성합니다. KoBERT 실험은 확장 실험으로 보존하지만, 팀 발표의 메인 파이프라인에는 포함하지 않습니다.

리뷰 텍스트와 target 평균 별점은 같은 강의의 기존 리뷰에서 나온 정보입니다. 따라서 이 결과는 리뷰가 이미 존재하는 강의의 품질 추정에는 사용할 수 있지만, 리뷰가 없는 신규 강의의 cold-start 성능을 의미하지는 않습니다.

## Graph-augmented MLP

- Node: 강의
- Node feature: 강의별 31차원 vector
- Edge: cosine similarity가 높은 k개의 강의를 이웃으로 연결
- Input: 자기 feature, 이웃 평균 feature, 두 feature의 차이

실제 학생-강의 interaction이 없는 상태에서는 Graph-augmented MLP가 Ridge와 KNN보다 낮은 성능을 보였습니다. 따라서 현재 데이터에서는 복잡한 그래프 구조의 이점이 확인되지 않았습니다.

## Top-K Recommendation

학생 persona는 강의와 동일한 feature space의 preference vector로 표현됩니다. 학생을 그래프 노드로 학습하는 대신 추천 시점의 query vector로 사용합니다.

```text
recommendation_score
  = 0.7 * preference_similarity
  + 0.3 * predicted_quality
```

- `preference_similarity`: 학생 선호와 강의 vector의 cosine similarity를 0~1로 변환
- `predicted_quality`: Ridge regression이 예측한 강의 품질
- K: 기본값 10

지원하는 persona preset:

- `low_workload`
- `learning_quality`
- `exam_light`
- `no_team_project`
- `challenging_but_good`

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

전처리된 feature 데이터로 전체 CV 실험:

```powershell
python scripts\run_full_cv_experiment.py
python scripts\make_cv_visuals.py
python scripts\run_extended_model_experiment.py
python scripts\run_leakage_safe_legacy_models.py
python scripts\make_extended_model_visuals.py
python scripts\build_transformer_embeddings.py
python scripts\run_transformer_tabular_experiment.py
python scripts\run_transformer_ablation.py
python scripts\make_transformer_visuals.py
python scripts\run_kobert_all_models.py
python scripts\make_kobert_all_models_visual.py
python scripts\run_kobert_mlp_experiment.py
python scripts\make_six_model_comparison.py
```

Top-10 추천:

```powershell
python scripts\recommend_topk.py --preset low_workload --top-k 10
```

직접 persona 가중치 입력:

```powershell
python scripts\recommend_topk.py --weights-json '{ "assignment_low_score": 1.0, "exam_light_score": 0.8, "text_positive_ratio": 0.6 }'
```

## Repository Structure

```text
scripts/                  preprocessing, experiments, recommendation
data/model/               anonymized lecture feature vectors
data/experiments/full_cv/ cross-validation results and figures
data/recommendations/     recommendation descriptions and summaries
```

## Limitations

- 과목명, 교수명, 학과 metadata가 없어 결과가 `lecture_id` 중심으로 표시됩니다.
- 사용자 ID와 학생별 수강/평가 이력이 없어 진정한 collaborative filtering은 수행하지 못했습니다.
- 현재 평가는 평균 평점 예측 성능을 중심으로 하며, 개인화 추천 성능 자체를 검증한 것은 아닙니다.
- 실제 개인화 평가에는 학생 interaction 데이터와 Hit@K, Recall@K, NDCG@K 등의 ranking metric이 필요합니다.

## Data Notice

원본 강의평 JSON과 리뷰 텍스트는 개인정보, 서비스 약관 및 저작권 문제를 고려해 공개 저장소에서 제외합니다. 저장소에는 실험 재현에 필요한 코드와 익명화된 파생 feature 및 결과만 포함합니다.
