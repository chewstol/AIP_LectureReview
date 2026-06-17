## 데이터 설명

- 원본 강의평: 14,056개
- 강의별 상세 요약: 2,451개
- 리뷰 5개 이상인 강의 노드: 753개
- 실험용 10% 샘플: 76개 강의 노드
- Train/Test 분할: 60개 / 16개

## 강의평 원문 사용 방식

`lecture_articles.csv`의 강의평 본문을 사용했다. 텍스트는 간단한 토큰화 후 TF-IDF 방식으로 단어 중요도를 계산했고, 주요 키워드를 아래 카테고리로 매핑했다.

- 과제: 과제, 레포트, 숙제, 제출, 퀴즈 등
- 시험: 시험, 중간, 기말, 오픈북, 족보, 암기 등
- 조모임: 팀플, 조별, 발표, 프로젝트 등
- 출결: 출석, 출결, 결석, 지각, 호명 등
- 성적: 학점, 성적, 후함, 깐깐, 절평, 상평 등
- 강의력: 설명, 강의력, 전달, 이해, 재미, 지루 등

감성 정보는 강의평 별점을 사용해 4-5점은 긍정, 1-2점은 부정, 3점은 중립으로 라벨링했다. 이후 카테고리별 TF-IDF, 긍정 TF-IDF, 부정 TF-IDF를 강의 단위로 평균 집계했다.

## 사용 Feature

- 정형 feature: 과제 적음/많음, 조모임 적음/많음, 성적 너그러움/깐깐함, 출결 부담, 시험 부담
- 텍스트 feature: 카테고리별 TF-IDF, 긍정 TF-IDF, 부정 TF-IDF, 긍정/부정/중립 비율
- 총 feature 수: 31개
- Target: `rating_average_norm`

## 베이스라인 및 제안 모델

- Baseline 1: Train 평균 평점 예측
- Baseline 2: Ridge Regression
- Baseline 3: Content KNN
- Proposed Model: Graph-augmented MLP

Graph-augmented MLP는 각 강의 feature와 유사 강의 이웃 feature 평균을 함께 입력으로 사용한다.

## 성능 비교

| Model | Test MSE | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: |
| Train mean rating | 0.01273292 | 0.11284025 | 0.09234167 |
| Ridge regression | 0.00550202 | 0.07417562 | 0.05714681 |
| Content KNN | 0.00738258 | 0.08592196 | 0.06305195 |
| Graph-augmented MLP best epoch | 0.05347417 | 0.23124483 | 0.16574160 |
| Graph-augmented MLP final epoch | 0.06273821 | 0.25047597 | 0.17400972 |

텍스트 feature를 포함하자 Ridge Regression과 Content KNN 성능이 정형 feature만 사용했을 때보다 개선되었다. 10% 샘플에서는 단순 모델이 더 안정적이었고, 복잡한 MLP는 데이터가 적어 과적합되는 모습을 보였다.

## Overfitting 관측

- Best epoch: 32
- Best epoch train loss: 0.04109466
- Best epoch test loss: 0.05347417
- Final epoch: 800
- Final epoch train loss: 0.00000103
- Final epoch test loss: 0.06273821

Epoch을 늘리자 train loss는 거의 0에 수렴했지만, test loss는 best epoch 이후 증가했다. 이는 작은 데이터에서 복잡한 모델이 학습 데이터에 과하게 맞춰져 일반화 성능이 떨어지는 과적합 현상으로 해석할 수 있다.

## 해석

이번 10% 실험은 최종 모델 성능 확보보다 제안 파이프라인 검증 목적이다. 강의평 원문 TF-IDF, 감성 라벨링, 강의 노드 벡터 생성, 베이스라인 비교, 제안 모델 학습, 과적합 관측까지 구현했다. 현재 결과에서는 단순한 Ridge Regression과 Content KNN이 더 좋은 성능을 보여, 다음 단계에서는 더 많은 데이터 사용, feature 정제, early stopping, regularization, 학생 개인 평가 이력 기반 선호 벡터 학습이 필요하다.
