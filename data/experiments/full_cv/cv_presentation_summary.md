# 06/08 전체 데이터 기반 검증 요약

## 실험 목적

전체 강의 노드 753개를 사용해 제안한 강의 특성 벡터 기반 모델을 검증했다. 이번 실험의 target은 `rating_average_norm`이며, 이는 강의 평균 평점을 0~1 범위로 정규화한 값이다. 즉, 강의평 텍스트와 정형 평가 feature가 강의 만족도를 얼마나 잘 설명하는지 평가했다.

## 데이터 및 Feature

- 전체 강의 노드: 753개
- Cross-validation: 5-fold
- 입력 feature: 31개
- Target: `rating_average_norm`

Feature는 두 종류로 구성했다.

- 정형 feature: 과제 적음/많음, 조모임 적음/많음, 성적 너그러움/깐깐함, 출결 부담, 시험 부담
- 텍스트 feature: 강의평 원문 TF-IDF, 카테고리별 긍정/부정 TF-IDF, 별점 기반 긍정/중립/부정 비율

## 비교 모델

- Train Mean: 학습 데이터 평균 평점을 예측하는 최소 기준 모델
- Ridge Regression: 정규화가 적용된 선형 회귀 baseline
- Content KNN: cosine similarity 기반 유사 강의 평점 예측 baseline
- Graph-augmented MLP: 강의 feature와 유사 강의 이웃 feature 평균을 함께 사용하는 제안 모델

## Cross-validation 하이퍼파라미터 탐색

5-fold cross-validation으로 각 모델의 하이퍼파라미터를 탐색했다.

- Ridge Regression: `alpha = 0, 0.001, 0.01, 0.1, 1, 10, 100`
- Content KNN: `k = 1, 3, 5, 7, 10, 15, 20, 30`
- Graph-augmented MLP:
  - hidden_dim=16, lr=0.01, epochs=300
  - hidden_dim=32, lr=0.01, epochs=300
  - hidden_dim=64, lr=0.03, epochs=800

## 주요 성능 결과

| Model | Best Params | CV MSE | CV RMSE | CV MAE |
| --- | --- | ---: | ---: | ---: |
| Ridge Regression | alpha=10.0 | 0.00319527 | 0.05641645 | 0.04178956 |
| Content KNN | k=20 | 0.00419486 | 0.06468836 | 0.04786706 |
| Train Mean | none | 0.01308109 | 0.11419993 | 0.09031465 |
| Graph-augmented MLP | hidden_dim=64, lr=0.03, epochs=800 | 0.01906956 | 0.13316284 | 0.09977101 |

가장 좋은 성능은 Ridge Regression이 보였다. Content KNN도 Train Mean보다 확실히 좋은 성능을 보였으므로, 강의 feature가 유사한 강의는 평균 평점도 유사하다는 가설을 어느 정도 지지한다.

## 의미 있는 결과 관측

1. TF-IDF와 정형 feature를 결합한 강의 벡터는 평균 평점 예측에 유효했다.
2. Ridge Regression이 가장 좋은 성능을 보였다. 현재 feature와 평균 평점 사이 관계는 복잡한 비선형 모델보다 선형 모델로도 잘 설명된다.
3. KNN은 k가 너무 작을 때보다 k=15~20 근처에서 더 안정적이었다. 이는 너무 적은 이웃만 보면 noise에 민감하고, 적절한 이웃 수가 필요함을 보여준다.
4. Graph-augmented MLP는 Train Mean보다는 일부 설정에서 개선될 수 있지만, Ridge/KNN보다 낮은 성능을 보였다. 현재 그래프는 실제 학생-강의 관계가 아니라 feature 유사도 기반 이웃이므로, 그래프 모델의 장점이 충분히 드러나기 어렵다.

## Qualitative Analysis

학생 선호 시나리오를 세 가지로 구성해 Top-10 추천 후보를 확인했다.

- low_workload: 과제 적음, 팀플 적음, 성적 너그러움, 시험 부담 낮음 선호
- learning_quality: 강의력/설명 관련 긍정 텍스트가 많은 강의 선호
- avoid_team_project: 팀플/프로젝트 부담을 피하는 선호

결과 파일 `qualitative_top10_scenarios.csv`에는 각 시나리오별 Top-10 lecture_id와 similarity, 추천 score가 저장되어 있다. 현재 데이터에는 과목명/교수명 metadata가 없어 정성 분석은 lecture_id 단위로 진행했다. 실제 서비스 적용을 위해서는 lecture_id와 과목명/교수명을 연결하는 metadata table이 필요하다.

## 제안 모델의 장점

- 강의평 원문과 정형 평가 데이터를 함께 사용한다.
- TF-IDF 기반 feature는 해석 가능성이 높다.
- 학생 선호 벡터와 강의 벡터 간 유사도 계산으로 Top-K 추천으로 확장하기 쉽다.
- Graph-augmented 구조는 향후 학생-강의-태그 관계 데이터가 추가될 때 GNN으로 확장 가능하다.

## 제안 모델의 단점 및 한계

- 현재는 학생별 수강/평가 이력이 없어 진짜 개인화 추천 성능을 직접 검증하지 못했다.
- 현재 target은 평균 평점 예측이므로, 추천 시스템 전체를 검증한 것은 아니다.
- 과목명/교수명 metadata가 없어 추천 결과를 사람이 해석하기 어렵다.
- Graph-augmented MLP는 현재 데이터 조건에서는 Ridge/KNN보다 낮은 성능을 보였다.
- 감성 라벨은 별점 기반 약지도 라벨이므로 실제 문장 감성과 다를 수 있다.

## 다음 단계

- lecture_id와 과목명/교수명 metadata 수집
- 학생별 강의평/수강 이력 기반 선호 벡터 생성
- Hit@K, NDCG@K 같은 추천 지표로 개인화 추천 평가
- TF-IDF 외 sentence embedding, KoBERT 등 텍스트 feature 개선
- 실제 학생-강의-태그 그래프를 구성한 뒤 GNN 적용
