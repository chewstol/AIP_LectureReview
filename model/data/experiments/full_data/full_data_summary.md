# 전체 데이터 실험 결과 요약

## 실험 설정

- 사용 노드: 리뷰 5개 이상 강의 노드 753개 전체
- Train/Test 분할: 602개 / 151개
- 입력 feature: 정형 feature + TF-IDF/감성 텍스트 feature 총 31개
- Target: `rating_average_norm`
- 평가 지표: MSE, RMSE, MAE

## 성능 비교

| Model | Test MSE | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: |
| Train mean rating | 0.01229654 | 0.11088977 | 0.09182209 |
| Ridge regression | 0.00263253 | 0.05130820 | 0.03944364 |
| Content KNN | 0.00357093 | 0.05975729 | 0.04606185 |
| Graph-augmented MLP best epoch | 0.02099083 | 0.14488211 | 0.10858397 |
| Graph-augmented MLP final epoch | 0.02099083 | 0.14488211 | 0.10858397 |

## 10% 실험 대비 변화

- Ridge Regression: MSE 0.00550202 -> 0.00263253
- Content KNN: MSE 0.00738258 -> 0.00357093
- Graph-augmented MLP: MSE 0.05347417 -> 0.02099083

전체 데이터를 사용하면서 모든 주요 모델의 test error가 감소했다. 특히 Ridge와 KNN은 10% 실험 대비 성능이 크게 개선되었고, MLP도 데이터 증가에 따라 성능이 좋아졌다.

## 해석

전체 데이터에서도 가장 좋은 모델은 Ridge Regression이었다. 이는 현재 feature와 평균 평점 사이의 관계가 복잡한 비선형 모델 없이도 상당 부분 선형적으로 설명된다는 뜻으로 볼 수 있다.

Content KNN도 안정적인 성능을 보였다. 강의 feature가 유사한 강의끼리는 평균 평점도 어느 정도 유사하다는 가설을 지지한다.

Graph-augmented MLP는 10% 실험보다 개선되었지만, 여전히 Ridge/KNN보다 낮은 성능이다. 데이터가 753개로 늘었지만 MLP를 안정적으로 학습하기에는 여전히 표본 수가 제한적이고, 현재 graph augmentation이 실제 학생-강의 관계를 반영하지 못하고 feature 유사도 기반 이웃만 사용하기 때문으로 해석할 수 있다.

## Baseline 진단

Ridge는 `alpha=1.0`에서 test MSE 0.00263253으로 가장 좋았다. 전체 데이터에서는 alpha 변화에 따른 성능 차이가 크지 않아, 정규화 강도에 비교적 안정적인 모습을 보였다.

KNN은 `k=15`에서 test MSE 0.00321814로 가장 좋았다. k가 너무 작으면 일부 이웃에 민감하고, 적절히 늘리면 더 안정적인 예측이 가능했다.

## 결론

전체 데이터를 사용한 결과, 강의평 텍스트/정형 feature 기반 강의 벡터가 평균 평점을 예측하는 데 유의미하게 작동함을 확인했다. 다만 현재 조건에서는 복잡한 Graph-augmented MLP보다 Ridge Regression과 Content KNN이 더 안정적이다. 향후 실제 학생별 수강/평가 이력과 과목명/교수명 metadata가 확보되면, 개인화 추천과 GNN 기반 모델을 더 정당하게 검증할 수 있다.
